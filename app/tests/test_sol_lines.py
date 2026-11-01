"""
Sol decides, and reading ahead: the tests of the arms sol, mix-lines, luna-lines and sol-lines.

    python -m pytest tests/test_sol_lines.py -q -p no:cacheprovider

Sol (luna_player.MODELS["sol"]) is asked through OpenAI's Responses API: the same brief and the
same choices as Luna's question (responses_body), its answer read into the Decisions shape
(responses_answers). The lines are the search's favourite actions played forward a few turns in
the helper (bench/fp_bridge_server.py "lines"), put on our 14 actions (foulplay_bridge._fill_lines),
compared by the arm mix-lines (search_player.lines_pick) and written into a model's brief
(luna_player._look_ahead).

No key of OpenAI's is involved: the fake server of tests/fake_decisions.py has a key of its own.
The player tests need what tests/test_search_player.py needs; the helper's own test needs Foul
Play (it is skipped where there is none).
"""

from __future__ import annotations

import importlib.util
import io
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

import foulplay_bridge as fb  # noqa: E402
import ladder  # noqa: E402
import luna_player as lp  # noqa: E402
import luna_report  # noqa: E402
import search_player as sp  # noqa: E402
from fake_decisions import FakeDecisions, check_responses_request, questions_of_responses  # noqa: E402

_spec = importlib.util.spec_from_file_location("tl_for_sol", os.path.join(REPO, "tests", "test_luna.py"))
tl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tl)
tsp = tl.tsp
classes = tl.classes                         # the fixture: the search player with the stub network
FAKE_KEY = tl.FAKE_KEY
SOL = lp.MODELS["sol"]


def sol_client(fake, effort="none", **kw):
    said = []
    kw.setdefault("say", said.append)
    client = lp.LunaClient(FAKE_KEY, url=fake.responses_url, model=SOL.model, api="responses", effort=effort, label="Sol", spec=SOL, **kw)
    client.said = said
    return client


def sol_built(n=3, mix=0):
    built = tl.toy_built(n, mix)
    built.body["model"] = SOL.model
    return built


def sol_for(fake, **settings):
    settings.setdefault("timeout", 0.6)
    return lp.Luna(sol_client(fake), lp.LunaSettings(**settings), spec=SOL)


# =============================================================================================
# The request and the answer of the Responses API
# =============================================================================================

def test_sols_request_asks_the_same_question_in_the_responses_shape():
    built = sol_built(3, mix=1)
    body = lp.responses_body(built.body, "none")
    assert check_responses_request(body) == []
    assert body["model"] == "gpt-6-sol" and body["reasoning"] == {"effort": "none"} and body["store"] is False
    assert body["max_output_tokens"] == 400 and lp.responses_body(built.body, "low")["max_output_tokens"] == 6000
    system, user = body["input"]
    assert system["role"] == "system" and system["content"].startswith("Pick one.") and lp.RESPONSES_SYSTEM in system["content"]
    assert user["role"] == "user" and user["content"].startswith("A toy position.")
    assert "- a1-toy: Choice 1. net 10%. MIX'S PICK." in user["content"]
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    action = fmt["schema"]["properties"]["action"]
    assert action["properties"]["choice"]["enum"] == ["a0-toy", "a1-toy", "a2-toy"] and action["required"] == ["reason", "choice", "confidence"]
    # the fake reads the question back out of it: the same values, the same descriptions
    (q,) = questions_of_responses(body)
    assert q["name"] == "action" and [c["value"] for c in q["choices"]] == ["a0-toy", "a1-toy", "a2-toy"]
    assert q["choices"][1]["description"] == "Choice 1. net 10%. MIX'S PICK."
    # the check's two questions: a predicate and a choice, each named in the schema
    check = lp.responses_body(lp.check_body("gpt-6-sol"), "none")
    assert check_responses_request(check) == [] and sorted(check["text"]["format"]["schema"]["properties"]) == ["sum", "two_plus_two"]
    assert "QUESTION \"two_plus_two\" (yes or no): Is it true that 2 + 2 = 4?" in check["input"][1]["content"]
    assert check["input"][0]["content"] == lp.RESPONSES_SYSTEM


def reply_with(text=None, refusal=False, **more):
    content = []
    if text is not None:
        content.append({"type": "output_text", "text": text})
    if refusal:
        content.append({"type": "refusal", "refusal": "no"})
    out = {"status": "completed", "output": [{"type": "reasoning"}, {"type": "message", "content": content}]}
    out.update(more)
    return out


def test_sols_answer_is_read_into_the_decisions_shape():
    body = sol_built(3).body
    answers, why = lp.responses_answers(reply_with(json.dumps({"action": {"reason": "It wins.", "choice": "a2-toy", "confidence": 0.7}})), body)
    assert why == "" and answers == [{"type": "choice", "name": "action", "choice": "a2-toy", "confidence": 0.7, "probabilities": [],
                                      "reason": "It wins."}]
    parsed = lp.parse_answer(answers, "action", {"a0-toy": 0, "a1-toy": 1, "a2-toy": 2})
    assert parsed["kind"] == "choice" and parsed["action"] == 2 and parsed["confidence"] == 0.7 and parsed["reason"] == "It wins."
    assert parsed["probabilities"] == {}
    answers, _ = lp.responses_answers(reply_with(refusal=True), body)
    assert answers == [{"type": "refusal", "name": "action"}] and lp.parse_answer(answers, "action", {"a0-toy": 0})["kind"] == "refusal"
    for data, said in ((reply_with(), "no output text"), (reply_with("not json {"), "not JSON"), (reply_with("[1]"), "not a JSON object"),
                       (reply_with(json.dumps({"other": {}})), "no answer named action"), ("nope", "not a JSON object"),
                       ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}, "max_output_tokens")):
        answers, why = lp.responses_answers(data, body)
        assert answers is None and said in why, (data, why)
    # the SDK's convenience field is read when there is no message item
    answers, _ = lp.responses_answers({"output": [], "output_text": json.dumps({"action": {"choice": "a0-toy"}})}, body)
    assert answers[0]["choice"] == "a0-toy"
    # a choice outside the offered values is told apart later, as Luna's is
    answers, _ = lp.responses_answers(reply_with(json.dumps({"action": {"reason": "", "choice": "zz", "confidence": 2}})), body)
    assert lp.parse_answer(answers, "action", {"a0-toy": 0})["kind"] == "illegal"


def test_the_prices_of_the_two_models():
    assert lp.cost_usd(SOL, 2500, 60) == pytest.approx(2500 * 2.0 / 1e6 + 60 * 10.0 / 1e6)
    assert lp.cost_usd(SOL, 2500, 60, cached_tokens=2000) == pytest.approx((500 * 2.0 + 2000 * 0.20 + 60 * 10.0) / 1e6)
    assert lp.cost_usd(lp.MODELS["luna"], 2500, 60) == pytest.approx(2500 * 0.10 / 1e6)          # Luna: input only
    assert lp.cost_usd(None, 1000) == pytest.approx(0.0001) and lp.cost_usd(SOL, None, None) == 0.0
    assert (SOL.api, SOL.model, SOL.effort, SOL.url) == ("responses", "gpt-6-sol", "none", "https://api.openai.com/v1/responses")
    assert lp.endpoint_problem(SOL.url) == ""


# =============================================================================================
# The client against the fake Responses API
# =============================================================================================

def test_sol_answers_through_the_responses_api():
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        client = sol_client(fake)
        built = sol_built(3, mix=2)
        reply = client.ask_sync(built.body, 3.0)
        client.close()
    parsed = lp.parse_answer(reply.answers, "action", built.values)
    assert parsed["kind"] == "choice" and parsed["action"] == 2 and parsed["reason"] == "The fake's reason for a2-toy."
    assert reply.input_tokens > 50 and reply.output_tokens > 5 and not reply.reasoning_tokens and reply.cached_tokens == 0
    (r,) = fake.requests
    assert r["problems"] == [] and r["api"] == "responses" and r["effort"] == "none" and r["path"] == "/v1/responses"
    assert r["authorization"] == "ok" and FAKE_KEY not in json.dumps(r)


@pytest.mark.parametrize("behaviour, kind, status, requests_seen", [
    ("http_401", "http", 401, 1), ("http_429", "http", 429, 1), ("http_500", "http", 500, 2), ("malformed_json", "malformed", 200, 1),
    ("malformed_shape", "malformed", 200, 1), ("incomplete", "malformed", 200, 1), ("html", "http", 502, 2), ("reset", "network", None, 2),
])
def test_every_failure_of_the_responses_api_is_an_error_without_the_key(behaviour, kind, status, requests_seen):
    with FakeDecisions(key=FAKE_KEY, mode=behaviour) as fake:
        client = sol_client(fake)
        with pytest.raises(lp.LunaError) as failed:
            client.ask_sync(sol_built().body, 3.0)
        client.close()
    assert failed.value.kind == kind and failed.value.status == status and len(fake.requests) == requests_seen
    assert FAKE_KEY not in failed.value.text and FAKE_KEY not in " ".join(client.said)


def test_a_slow_sol_is_given_up_at_the_timeout_and_a_refusal_is_not_an_error():
    with FakeDecisions(key=FAKE_KEY, mode="hang", hang_s=1.5) as fake:
        client = sol_client(fake)
        began = time.monotonic()
        with pytest.raises(lp.LunaError) as slow:
            client.ask_sync(sol_built().body, 0.25)
        assert slow.value.kind == "timeout" and time.monotonic() - began < 0.9
        fake.mode = "refusal"
        reply = client.ask_sync(sol_built().body, 3.0)
        assert lp.parse_answer(reply.answers, "action", sol_built().values)["kind"] == "refusal"
        fake.mode = "illegal"
        reply = client.ask_sync(sol_built().body, 3.0)
        assert lp.parse_answer(reply.answers, "action", sol_built().values)["kind"] == "illegal"
        client.close()


def test_an_effort_the_api_refuses_falls_back_to_low_once():
    with FakeDecisions(key=FAKE_KEY, mode="mix", refuse_efforts=("none",)) as fake:
        client = sol_client(fake)
        first = client.ask_sync(sol_built(3, mix=1).body, 3.0)
        second = client.ask_sync(sol_built(3, mix=0).body, 3.0)
        client.close()
    assert lp.parse_answer(first.answers, "action", sol_built().values)["action"] == 1 and first.attempts == 1
    assert lp.parse_answer(second.answers, "action", sol_built().values)["action"] == 0
    assert [r["effort"] for r in fake.requests] == ["none", "low", "low"] and [r["status"] for r in fake.requests] == [400, 200, 200]
    assert client.effort == "low" and client.effort_fell_back == "none"
    assert len(client.said) == 1 and "refused reasoning effort \"none\"" in client.said[0] and "low" in client.said[0]
    assert first.reasoning_tokens == 96 and first.output_tokens > 96       # with reasoning: its tokens are output tokens, and are paid for
    # every effort refused: the error stands after the one fallback, and nothing loops
    with FakeDecisions(key=FAKE_KEY, mode="mix", refuse_efforts=("none", "low")) as fake:
        client = sol_client(fake)
        with pytest.raises(lp.LunaError) as refused:
            client.ask_sync(sol_built().body, 3.0)
        client.close()
    assert refused.value.status == 400 and len(fake.requests) == 2


def test_sols_choice_carries_its_reason_its_tokens_and_its_cost():
    import asyncio

    with FakeDecisions(key=FAKE_KEY, mode="other") as fake:
        sol = sol_for(fake, guard=False)
        built, facts = sol_built(3, mix=0), tl.toy_facts(3, mix=0)
        action, info = asyncio.run(sol.choose(built, facts, 5.0))
        sol.stamp(info)
        sol.client.close()
    assert action == 1 and info["choice"] == 1 and info["mix"] == 0 and not info["fell_back"]
    assert info["reason"] == "The fake's reason for a1-toy." and info["who"] == "sol" and info["model"] == "gpt-6-sol"
    assert info["tokens_from"] == "usage" and info["output_tokens"] > 0
    assert info["cost_usd"] == pytest.approx(lp.cost_usd(SOL, info["input_tokens"], info["output_tokens"]), abs=1e-7)
    counts = lp.new_counts()
    lp.add_counts(counts, info)
    fields = lp.result_fields(counts)
    assert fields["luna_model"] == "gpt-6-sol" and fields["luna_output_tokens"] == info["output_tokens"]
    assert fields["luna_cost_usd"] == pytest.approx(info["cost_usd"], abs=1e-6)
    text = lp.counts_text(counts, "Sol")
    assert text.startswith("Sol: asked on 1 of 1") and "output tokens" in text
    # Luna's counts say what they said before the models: input tokens at Luna's price, no output
    old = lp.new_counts()
    lp.add_counts(old, {"asked": True, "input_tokens": 2000, "choice": 0, "mix": 0, "latency_ms": 100.0})
    assert lp.result_fields(old)["luna_cost_usd"] == pytest.approx(0.0002) and "output tokens" not in lp.counts_text(old)
    assert lp.counts_text(old).startswith("Luna: asked on 1 of 1")


def test_the_check_for_sol_says_its_latency_its_reason_and_its_cost():
    out = []
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        client = sol_client(fake)
        assert lp.luna_check(client, game_timeout=5.0, out=out.append) == 0
        client.close()
    text = "\n".join(out)
    assert out[0].startswith("Sol check: one tiny question to http://127.0.0.1") and "model gpt-6-sol" in out[0] and "key ***" in out[0]
    assert "predicate two_plus_two" in text and "choice sum (what is 2+2): four" in text and "NO probabilities" not in text
    assert "its reason: The fake's reason" in text and "tokens:" in text and "a decision" in text
    assert out[-1].startswith("Sol check OK") and FAKE_KEY not in text
    assert all(r["problems"] == [] for r in fake.requests) and len(fake.requests) == 3
    out = []
    with FakeDecisions(key=FAKE_KEY, mode="mix", refuse_efforts=("none",)) as fake:
        client = sol_client(fake)
        assert lp.luna_check(client, game_timeout=5.0, out=out.append) == 0
        client.close()
    assert any("refused reasoning effort \"none\"; \"low\" is used" in line for line in out)
    out = []
    with FakeDecisions(key="sk-fake-another-key-000000", mode="mix") as fake:
        client = sol_client(fake)
        assert lp.luna_check(client, out=out.append) == 1
        client.close()
    assert len(out) == 2 and out[1].startswith("SOL CHECK FAILED: HTTP 401") and FAKE_KEY not in " ".join(out)
    # Luna's check reads as it did
    out = []
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        client = tl.client_for(fake)
        assert lp.luna_check(client, out=out.append) == 0
        client.close()
    assert out[0].startswith("Luna check: one tiny question") and out[-1].startswith("Luna check OK") and "probabilities three" in "\n".join(out)


# =============================================================================================
# ladder.py: the arms, the flags, the budgets, the key
# =============================================================================================

def test_the_new_arms_are_modes_of_the_player_and_of_ladder():
    for arm in ("sol", "mix-lines", "luna-lines", "sol-lines"):
        assert arm in sp.MODES and arm in ladder.PLAYER_MODES and sp.split_mode(arm) == ("mix", None)
    assert sp.DECIDERS == {"luna": "luna", "sol": "sol", "luna-lines": "luna", "sol-lines": "sol", "luna-solo": "luna", "sol-solo": "sol",
                           "fable-solo": "fable", "opus-solo": "opus", "sonnet-solo": "sonnet", "haiku-solo": "haiku"}
    assert sp.LINES_ARMS == ("mix-lines", "luna-lines", "sol-lines")
    assert ladder.decider_models(ladder.parse_args(["--player", "sol"])) == ["sol"]
    assert ladder.decider_models(ladder.parse_args(["--arms", "mix,sol-lines,luna-lines"])) == ["luna", "sol"]
    assert ladder.decider_models(ladder.parse_args(["--player", "mix-lines"])) == [] and not ladder.wants_luna(ladder.parse_args(["--player", "mix-lines"]))
    assert ladder.wants_luna(ladder.parse_args(["--player", "sol-lines"]))
    args = ladder.parse_args(["--player", "sol"])
    assert (args.sol_model, args.sol_timeout, args.sol_decide_s, args.sol_effort, args.model, args.key_stdin) == ("gpt-6-sol", 5.0, 5.0, "none", "luna", False)
    assert (args.lines_ms, args.lines_k, args.lines_turns, args.lines_worlds, args.lines_margin, args.lines_threads) == (1500, 3, 5, 2, 0.10, 4)
    assert ladder.model_label("sol") == "Sol" and ladder.model_label("gpt-6-sol") == "Sol" and ladder.model_label(None) == "Luna"
    for bad in (["--lines-k", "0"], ["--lines-turns", "9"], ["--lines-ms", "50"], ["--sol-timeout", "0"], ["--sol-effort", "max"],
                ["--lines-threads", "9"]):
        with pytest.raises(SystemExit):
            ladder.parse_args(bad)


def test_the_budgets_of_the_arms_and_what_a_decision_may_take():
    args = ladder.parse_args(["--arms", "mix,luna,sol,mix-lines,luna-lines,sol-lines,mix-deep"])
    budgets = ladder.arm_budgets(args, args.arms)
    assert sorted(budgets) == ["luna-lines", "mix-deep", "mix-lines", "sol", "sol-lines"]       # mix and luna: today's budget, as before
    sol = budgets["sol"]
    assert (sol.search_ms, sol.worlds_x, sol.deadline, sol.decide_s, sol.lines_ms, sol.max_decision_s) == (500, 1, 2.5, 5.0, 0, 12.0)
    lines = budgets["mix-lines"]
    assert lines.deadline == pytest.approx(2.5 + 1.875 + 0.2, abs=0.011) and lines.decide_s == 0.0
    assert lines.lines_spec() == {"k": 3, "turns": 5, "ms": 1500.0, "worlds": 2, "threads": 4} and lines.lines_s() == 1.875
    assert budgets["luna-lines"].decide_s == pytest.approx(1.3) and budgets["sol-lines"].decide_s == 5.0
    assert budgets["sol-lines"].as_dict()["lines"] == lines.lines_spec() and budgets["sol-lines"].as_dict()["decide_s"] == 5.0
    assert budgets["mix-deep"] == ladder.deep_budget(args) and budgets["mix-deep"].lines_spec() is None
    assert ladder.search_problems(args) == []
    # the player's own defaults are the same budgets
    assert sp.ARM_BUDGETS["sol"] == sol and sp.ARM_BUDGETS["mix-lines"].lines_spec() == lines.lines_spec()
    assert sp.ARM_BUDGETS["mix-deep"] is sp.DEEP_BUDGET
    # a decision of 10 s or more costs two ticks of the battle timer: refused before anything starts
    slow = ladder.parse_args(["--player", "sol-lines", "--sol-decide-s", "6"])
    (problem,) = ladder.search_problems(slow)
    assert "sol-lines" in problem and "two ticks" in problem
    # a deadline that cannot hold the search and its lines
    with pytest.raises(ValueError):
        sp.SearchBudget(search_ms=500, deadline=2.5, lines_ms=1500)
    with pytest.raises(ValueError):
        sp.SearchBudget(search_ms=500, deadline=6.0, lines_ms=1500, lines_k=7)
    # remote helpers: the round trip counts against a lines arm's deadline too
    assert any("mix-lines" in p for p in ladder.search_problems(ladder.parse_args(["--player", "mix-lines"]), rtt_s=1.5))
    assert ladder.search_problems(ladder.parse_args(["--player", "mix-lines"]), rtt_s=0.1) == []


class _Stdin:
    def __init__(self, data: bytes):
        self.buffer = io.BytesIO(data)

    def isatty(self):
        return False


def test_the_key_comes_on_standard_input_for_the_window_app(monkeypatch):
    args = ladder.parse_args(["--player", "sol", "--key-stdin"])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "stdin", _Stdin((FAKE_KEY + "\r\n").encode()))
    assert ladder.openai_key(args) == FAKE_KEY and "OPENAI_API_KEY" not in os.environ
    # an empty line: the variable, which is taken out of the environment at once
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(sys, "stdin", _Stdin(b"\n"))
    assert ladder.openai_key(args) == FAKE_KEY and "OPENAI_API_KEY" not in os.environ
    # neither: no key, said in one line, and nothing is asked at a prompt (there is no terminal behind the app)
    monkeypatch.setattr(sys, "stdin", _Stdin(b""))
    with pytest.raises(lp.LunaError) as none:
        ladder.openai_key(args)
    assert none.value.kind == "key"
    # a typed key wins over the variable
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-environment-0000")
    monkeypatch.setattr(sys, "stdin", _Stdin((FAKE_KEY + "\n").encode()))
    assert ladder.openai_key(args) == FAKE_KEY and "OPENAI_API_KEY" not in os.environ


def test_a_run_with_both_models_checks_each_and_plays_nothing_when_one_fails(monkeypatch, capsys, tmp_path):
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        args = ladder.parse_args(["--arms", "luna,sol-lines", "--luna-url", fake.url, "--sol-url", fake.responses_url, "--sol-timeout", "3"])
        assert ladder.luna_ready(args) == 0 and "OPENAI_API_KEY" not in os.environ
        made = args._deciders
        assert sorted(made) == ["luna", "sol"] and args._luna is made["luna"]
        assert (made["sol"].label, made["sol"].client.api, made["sol"].client.model, made["sol"].settings.timeout) == ("Sol", "responses", "gpt-6-sol", 3.0)
        assert (made["luna"].label, made["luna"].client.api, made["luna"].settings.timeout) == ("Luna", "decisions", 1.2)
        out = capsys.readouterr().out
        assert "Luna check OK" in out and "Sol check OK" in out and FAKE_KEY not in out
        assert sorted({r["path"] for r in fake.requests}) == ["/v1/decisions", "/v1/responses"]
        for who in made.values():
            who.client.close()
        # the header's lines
        assert "the Responses API, reasoning effort none" in ladder.sol_settings_text(args, made["sol"].client)
        assert "5 turns" in ladder.lines_settings_text(args) and "0.1 or more better and in no set worse" in ladder.lines_settings_text(args)
        # Sol's check fails: nothing is played, Luna's client is closed again, the note says whose
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        fake.refuse_efforts = ("none", "low")
        args = ladder.parse_args(["--arms", "luna,sol", "--luna-url", fake.url, "--sol-url", fake.responses_url,
                                  "--exit-file", str(tmp_path / "exit.txt")])
        assert ladder.luna_ready(args) == 5 and getattr(args, "_deciders", None) is None
        out = capsys.readouterr().out
        assert "SOL CHECK FAILED: HTTP 400" in out and "Sol cannot be used" in out and "Nothing was played" in out and FAKE_KEY not in out
        assert open(tmp_path / "exit.txt", encoding="utf-8").read().startswith("luna: Sol's check failed")


def test_the_spending_cap_turns_the_models_off_for_the_run():
    class Who:
        def __init__(self, label):
            self.label = label
            self.client = type("C", (), {"disabled": None, "off_reason": "key_refused"})()

    said = []
    cap = ladder.SpendCap(0.02)
    whos = {"sol": Who("Sol"), "luna": Who("Luna")}
    cap.add({"luna": {"cost_usd": 0.012}}, whos, said.append)
    cap.add({"played": 3}, whos, said.append)                          # a decision without a model: nothing
    cap.add({"luna": {"cost_usd": None}}, whos, said.append)
    assert not cap.hit and said == [] and whos["sol"].client.disabled is None and cap.spent == pytest.approx(0.012)
    cap.add({"luna": {"cost_usd": 0.009}}, whos, said.append)
    assert cap.hit and len(said) == 1 and "Luna and Sol OFF for the rest of this run" in said[0] and "$0.02" in said[0]
    assert all(w.client.disabled and w.client.off_reason == "spend_cap" for w in whos.values())
    cap.add({"luna": {"cost_usd": 0.5}}, whos, said.append)
    assert len(said) == 1                                             # said once
    free = ladder.SpendCap(0)
    free.add({"luna": {"cost_usd": 999.0}}, whos, said.append)
    assert not free.hit and len(said) == 1                            # 0: no cap
    assert ladder.parse_args([]).max_model_usd == 20.0
    # the real client says why it is off: the rail's name in the decision's record
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        client = sol_client(fake)
        assert client.available() == (True, "")
        client.off_reason, client.disabled = "spend_cap", "the cap"
        assert client.available() == (False, "spend_cap")
        client.close()
        other = tl.client_for(fake)
        other.disabled = "HTTP 401"
        assert other.available() == (False, "key_refused")           # the key's refusal reads as before
        other.close()


def test_luna_check_takes_the_model(monkeypatch, capsys):
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        assert ladder.luna_check_only(ladder.parse_args(["--luna-check", "--model", "sol", "--sol-url", fake.responses_url])) == 0
        out = capsys.readouterr().out
        assert "Sol check OK" in out and FAKE_KEY not in out and all(r["path"] == "/v1/responses" for r in fake.requests)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(lp.getpass, "getpass", lambda prompt="": "")
        assert ladder.luna_check_only(ladder.parse_args(["--luna-check", "--model", "sol", "--sol-url", fake.responses_url])) == 5
        assert "SOL CHECK FAILED: no OpenAI API key" in capsys.readouterr().out


def test_main_reads_the_password_then_the_key_from_standard_input_and_tells_the_app(monkeypatch, capsys, tmp_path):
    """The window app's contract: line 1 the password, line 2 the key; a key that fails is one "@app" error line and exit code 5."""
    weights = tmp_path / "learner_update_000001.pt"
    weights.write_bytes(b"not a network: the run ends before it is read")
    monkeypatch.setenv("PSBOT_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PS_PASSWORD", raising=False)
    kept = (ladder.OUT, ladder.GRAPH, ladder.BATTLES)
    common = ["--server", "local:1", "--checkpoint", str(weights), "--out", str(tmp_path / "results.jsonl"), "--no-console-log",
              "--app-events", "--password-stdin", "--key-stdin", "--player", "sol"]
    try:
        with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
            # the second line is the key: here a wrong one, which the API refuses
            monkeypatch.setattr(sys, "stdin", _Stdin(b"the-showdown-password\nsk-wrong-key-000000000000\n"))
            assert ladder.main(common + ["--user", "SolTestOne", "--sol-url", fake.responses_url]) == 5
            out = capsys.readouterr().out
            assert "SOL CHECK FAILED: HTTP 401" in out and "the-showdown-password" not in out and "sk-wrong-key" not in out
            events = [json.loads(line[5:]) for line in out.splitlines() if line.startswith("@app ")]
            assert any(e["ev"] == "phase" and e["text"] == "checking the OpenAI key with Sol" for e in events)
            assert events[-1]["ev"] == "error" and events[-1]["text"].startswith("Sol's check failed") and "sk-" not in json.dumps(events)
            assert len(fake.requests) == 1 and fake.requests[0]["authorization"] == "wrong"
            # no second line and no variable: no key, and the password was not taken for one
            monkeypatch.setattr(sys, "stdin", _Stdin(b"the-showdown-password\n"))
            assert ladder.main(common + ["--user", "SolTestTwo", "--sol-url", fake.responses_url]) == 5
            out = capsys.readouterr().out
            assert "Sol cannot be used: no OpenAI API key" in out and len(fake.requests) == 1
    finally:
        ladder.APP_EVENTS = False
        ladder.OUT, ladder.GRAPH, ladder.BATTLES = kept


# =============================================================================================
# The lines: on our actions, compared, and told
# =============================================================================================

def line(choice, value, turns=5, end=None, steps=None, **more):
    out = {"choice": choice, "fp": choice, "value": value, "end": value if end is None else end, "turns": turns,
           "steps": steps if steps is not None else [], "searches": turns}
    out.update(more)
    return out


def test_the_lines_of_the_worlds_land_on_our_actions_weighted_by_the_worlds_chance():
    advice = fb._advice_from({"ok": True}, 0.0, 0)
    advice.lines = {"k": 3, "turns": 5, "ms": 1400.0, "worlds": [
        {"world": 0, "chance": 0.6, "lines": [line("move 1", 0.70), line("switch 3", 0.40, turns=4), line("move 9", 0.99)]},
        {"world": 2, "chance": 0.2, "lines": [line("move 1", 0.30, cut=True), line("move 2", None), line("switch 3", 0.60)]},
        {"world": 1, "chance": 0.0, "error": "RuntimeError: a panic", "lines": []},
        "not a world",
    ]}
    index = {"move 1": 0, "move 2": 1, "switch 3": 6}.get
    fb._fill_lines(advice, index)
    table = advice.lines_by_action
    assert sorted(table) == [0, 6]                                   # "move 9" is no action of ours, "move 2" has no value
    assert table[0]["value"] == pytest.approx(0.6 / 0.8 * 0.70 + 0.2 / 0.8 * 0.30) and table[0]["turns"] == 5
    assert [w["world"] for w in table[0]["worlds"]] == [0, 2] and table[0]["worlds"][1]["cut"] is True
    assert table[6]["value"] == pytest.approx(0.75 * 0.40 + 0.25 * 0.60) and table[6]["turns"] == 4       # the shortest of its lines
    # no chances given: the worlds count the same
    advice.lines = {"worlds": [{"world": 0, "lines": [line("move 1", 0.2)]}, {"world": 1, "lines": [line("move 1", 0.6)]}]}
    fb._fill_lines(advice, index)
    assert advice.lines_by_action[0]["value"] == pytest.approx(0.4)
    advice.lines = {"error": "boom", "worlds": []}
    fb._fill_lines(advice, index)
    assert advice.lines_by_action == {}


def rows(**values):
    return {int(a[1:]): {"value": v, "turns": 5} for a, v in values.items()}


def test_mix_lines_changes_the_move_only_for_a_clearly_better_line_the_search_rates():
    share = [0.50, 0.30, 0.05, 0.15] + [0.0] * 10
    # another action's line ends 0.12 better, and the search rates that action: it is played
    action, info = sp.lines_pick(0, share, rows(a0=0.50, a1=0.62, a3=0.55))
    assert action == 1 and info["changed"] and info["why"] == "line" and info["gain"] == pytest.approx(0.12) and info["mix"] == 0
    assert info["values"] == {"0": 0.50, "1": 0.62, "3": 0.55}
    # 0.07 better: under the margin (0.10), mix's pick stands
    assert sp.lines_pick(0, share, rows(a0=0.50, a1=0.57)) == (0, {"mix": 0, "played": 0, "changed": False, "why": "close",
                                                                 "values": {"0": 0.5, "1": 0.57}, "turns": {"0": 5, "1": 5}})
    # better on average, but worse than mix's pick in one of the two worlds read: one likely future against another, not played
    mixed = {0: {"value": 0.50, "turns": 5, "worlds": [{"world": 0, "value": 0.40}, {"world": 1, "value": 0.60}]},
             1: {"value": 0.65, "turns": 5, "worlds": [{"world": 0, "value": 0.80}, {"world": 1, "value": 0.50}]}}
    action, info = sp.lines_pick(0, share, mixed)
    assert action == 0 and info["why"] == "worlds_disagree" and not info["changed"]
    assert sp.lines_pick(0, share, mixed, sp.LinesRule(agree=False))[0] == 1
    mixed[1]["worlds"][1]["value"] = 0.61                    # better in both worlds
    assert sp.lines_pick(0, share, mixed)[0] == 1
    mixed[1]["worlds"] = [{"world": 2, "value": 0.65}]       # read in another world only: nothing to disagree with
    assert sp.lines_pick(0, share, mixed)[0] == 1
    # the better line belongs to an action the search does not rate (under a quarter of its largest share): not played
    action, info = sp.lines_pick(0, share, rows(a0=0.50, a2=0.90))
    assert action == 0 and info["why"] == "mix_best"
    # mix's pick is the best of the lines
    assert sp.lines_pick(1, share, rows(a0=0.50, a1=0.62))[1]["why"] == "mix_best"
    # nothing to compare with
    assert sp.lines_pick(0, share, {})[1]["why"] == "no_lines"
    assert sp.lines_pick(2, share, rows(a0=0.50, a1=0.90))[1]["why"] == "no_line_for_mix"
    short = rows(a0=0.50, a1=0.90)
    short[0]["turns"] = 1
    assert sp.lines_pick(0, share, short)[1]["why"] == "short"
    short = rows(a0=0.50, a1=0.90)
    short[1]["turns"] = 1                       # the better line was cut short: not compared
    assert sp.lines_pick(0, share, short)[0] == 0
    # the rule's numbers
    assert sp.lines_pick(0, share, rows(a0=0.50, a1=0.57), sp.LinesRule(margin=0.05))[0] == 1
    assert (sp.LinesRule().margin, sp.LinesRule().keep, sp.LinesRule().min_turns, sp.LinesRule().agree) == (0.10, 0.25, 2, True)
    assert sp.lines_pick(0, share, rows(a0=0.50, a2=0.90), sp.LinesRule(keep=0.05))[0] == 2


def synthetic_state(ours, theirs):
    """An engine state text with the fields a line's snapshot reads: per side six Pokemon (id, hp, max hp, status), the active one, the boosts, the forced switch."""
    def mon(name, hp, maxhp, status="NONE"):
        f = [name, "80", "normal", "typeless", "normal", "typeless", str(hp), str(maxhp), "none", "none", "none", "serious", "85;85;85;85;85;85",
             "200", "200", "200", "200", "200", status, "0", "0", "50.0", "tackle;false;32", "none;false;32", "none;false;32", "none;false;32",
             "none", "false", "normal"]
        return ",".join(f)

    def side(mons, active, boosts=(0, 0, 0, 0, 0), must_switch=False):
        mons = list(mons) + [("none", 0, 0)] * (6 - len(mons))
        f = [mon(*m) for m in mons] + [str(active), ";".join(["0"] * 20), "NONE", "0;0;0;0", "0"] + [str(b) for b in boosts] + ["0", "0"] \
            + ["0", "0", "0", "0", "true" if must_switch else "false", "none", "false", "false", "false", "none"]
        return "=".join(f)

    return "/".join([side(**ours), side(**theirs), "none;0", "none;0", "false;0", "false"])


def test_a_lines_snapshot_reads_the_engines_state_text():
    server = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
        "fp_bridge_server_for_lines", os.path.join(REPO, "bench", "fp_bridge_server.py")))
    sys.modules["fp_bridge_server_for_lines"] = server
    server.__spec__.loader.exec_module(server)
    text = synthetic_state(
        dict(mons=[("garchomp", 150, 300, "BURN"), ("rotomwash", 0, 250), ("heatran", 280, 280)], active=0, boosts=(2, 0, 0, 0, -1)),
        dict(mons=[("skarmory", 0, 270), ("blissey", 651, 651), ("dragonite", 100, 300, "PARALYZE")], active=0, must_switch=True))
    snap = server._Server._snapshot(text)
    assert snap["ours"] == {"active": "garchomp", "hp": 50, "left": 2, "team_hp": round(430 / 830, 3), "status": "brn",
                            "boosts": {"atk": 2, "spe": -1}}
    assert snap["theirs"] == {"active": "skarmory", "hp": 0, "left": 2, "team_hp": round(751 / 1221, 3), "must_switch": True}
    # a decision with lines is not started when the search and the lines cannot end in time (the helper's own rule)
    made = object.__new__(server._Server)
    made.counters = {"expired": 0, "decide_errors": 0}
    seen = {}
    made.decide = lambda *a, **k: seen.update(k) or {"ok": True}
    now = time.monotonic()
    spec = {"k": 3, "turns": 5, "ms": 1500, "worlds": 2}
    late = made.handle({"op": "decide", "key": "k", "ms": 100, "lines": spec, "deadline": now + 1.0})
    assert late["ok"] is False and late["expired"] is True and "lines" not in seen
    ok = made.handle({"op": "decide", "key": "k", "ms": 100, "lines": spec, "deadline": now + 3.0})
    assert ok["ok"] is True and seen["lines"] == spec
    made.handle({"op": "decide", "key": "k", "ms": 100, "lines": "yes", "deadline": now + 3.0})
    assert seen["lines"] is None                       # not a specification: no lines, the search as ever


class _Names:
    @staticmethod
    def species_name(name):
        return {"garchomp": "Garchomp", "landorustherian": "Landorus-Therian", "heatran": "Heatran", "rotomwash": "Rotom-Wash"}.get(name, name.title())

    @staticmethod
    def move_name(name):
        return {"earthquake": "Earthquake", "outrage": "Outrage", "swordsdance": "Swords Dance"}.get(name, name.title())


class _Move:
    def __init__(self, name):
        self.id = name


def step(ours, theirs, our_side, their_side, **more):
    out = {"ours": ours, "theirs": theirs, "p": 0.9, "after": {"ours": our_side, "theirs": their_side}}
    out.update(more)
    return out


def test_the_look_ahead_section_tells_each_line_in_plain_words():
    ours = lambda hp, left=6, **m: dict({"active": "landorustherian", "hp": hp, "left": left, "team_hp": 0.8}, **m)   # noqa: E731
    foe = lambda name, hp, left=6, **m: dict({"active": name, "hp": hp, "left": left, "team_hp": 0.6}, **m)          # noqa: E731
    steps = [
        step("move earthquake", "move outrage", ours(41), foe("garchomp", 22)),
        step("move earthquake", "move outrage", ours(41), foe("garchomp", 0, 5, must_switch=True)),
        step("none", "switch heatran", ours(41), foe("heatran", 100, 5), forced=True),
        step("switch rotomwash", "move magmastorm", dict(active="rotomwash", hp=70, left=6, team_hp=0.75, status="brn"), foe("heatran", 100, 5)),
    ]
    facts = tl.toy_facts(3, mix=0)
    facts.actions = {0: (_Move("earthquake"), {}), 1: (_Move("swordsdance"), {"terastallize": True}), 2: (_Move("outrage"), {})}
    facts.lines = {
        0: {"value": 0.64, "end": 0.66, "turns": 3, "worlds": [{"world": 0, "chance": 0.6, "value": 0.71, "turns": 3, "steps": steps},
                                                               {"world": 1, "chance": 0.4, "value": 0.52, "turns": 3, "steps": []}]},
        1: {"value": 0.41, "end": 0.41, "turns": 5, "worlds": [{"world": 0, "chance": 0.6, "value": 0.41, "turns": 5, "steps": [
            step("move swordsdance terastallize", "move outrage", ours(12, boosts={"atk": 2}), foe("garchomp", 100))]}]},
        5: {"value": 0.99, "turns": 5, "worlds": []},                          # not an action of this request: left out
    }
    facts.revealed = ["garchomp"]
    board = type("Board", (), {"sm": _Names(), "facts": facts})()
    text = lp._look_ahead(board, 0)
    head, first, second = text.split("\n")
    assert head.startswith("LOOK-AHEAD LINES (our search played each of its favourite actions forward, up to 5 turns")
    assert first.startswith("- Earthquake: T1 we use Earthquake and foe uses Outrage -> our Landorus-Therian 41%, foe's Garchomp 22%. ")
    assert "T2 we use Earthquake and foe uses Outrage -> our Landorus-Therian 41%, foe's Garchomp fainted. " in first
    assert "then foe switches to Heatran -> our Landorus-Therian 41%, foe's Heatran (not shown yet: a guess) 100%. " in first
    assert "T3 we switch to Rotom-Wash and foe uses Magmastorm -> our Rotom-Wash 70% (burned), foe's Heatran (not shown yet: a guess) 100%. " in first
    assert first.endswith("END after 3 turns: we have 6 left (75% of our team's HP), foe 5 (60%). Value 0.64 (0.71, 0.52 in the sets read).")
    assert second.startswith("- Swords Dance + Terastallize: T1 we use Swords Dance and Terastallize and foe uses Outrage -> ")
    assert "our Landorus-Therian 12% (+2 Atk)" in second
    assert second.endswith("Value 0.41.")
    assert first.index("Earthquake") < len(first) and text.index("- Earthquake") < text.index("- Swords Dance")       # the better line first
    # shorter under the budget's pressure: three turns told, then the end alone, then nothing
    assert "T3" in lp._look_ahead(board, 1) and "T1" not in lp._look_ahead(board, 2) and "Value 0.64" in lp._look_ahead(board, 2)
    assert lp._look_ahead(board, 3) == ""
    facts.lines = {}
    assert lp._look_ahead(board, 0) == ""
    # a choice's own advisors line carries its line
    facts.lines = {0: {"value": 0.64, "turns": 3, "worlds": []}}
    facts.shares = [0.6, 0.3, 0.1] + [0.0] * 11
    assert ", line 0.64 after 3 turns" in lp._advisors(board, 0, facts.merged()) and "line" not in lp._advisors(board, 1, facts.merged())


# =============================================================================================
# The arms in the player (the stub network, the recorded search answers, the fake APIs)
# =============================================================================================

def with_lines(battles, values=(0.50, 0.70, 0.40), worlds=2):
    """
    The recorded battles with lines put into every recorded search answer: the search's three
    favourite options get a line each, valued `values` (the favourite's first), 5 turns long, in
    `worlds` worlds.
    """
    out = json.loads(json.dumps(battles))
    for events in out:
        for event in events:
            reply = event.get("fp_reply") if event.get("t") == "order" else None
            if not isinstance(reply, dict) or not reply.get("ok") or not reply.get("options"):
                continue
            ranked = sorted(reply["options"], key=lambda o: -float(o.get("policy") or 0.0))[:3]
            foe = {"active": "garchomp", "hp": 60, "left": 5, "team_hp": 0.55}
            mine = {"active": "heatran", "hp": 80, "left": 6, "team_hp": 0.7}
            reply["lines"] = {"k": 3, "turns": 5, "ms": 1234.5, "step_ms": 35.7, "candidates": [o.get("choice") for o in ranked], "worlds": [
                {"world": w, "chance": 1.0 / worlds, "lines": [
                    line(o.get("choice"), values[i], steps=[step("move earthquake", "move outrage", mine, foe)] * 5)
                    for i, o in enumerate(ranked)]} for w in range(worlds)]}
    return out


def play_with(classes, battles, mode, deciders=None, **kw):
    return tl.play_arm(classes, battles, mode, luna=None, deciders=deciders or {}, **kw)


@tsp.needs_player
def test_sol_answering_mixs_pick_plays_exactly_what_mix_plays(classes):
    battles = tsp.load_fixture()[:2]
    mix_notes, mix_orders, _ = tl.play_arm(classes, battles, "mix")
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        sol = sol_for(fake)
        notes, orders, players = play_with(classes, battles, "sol", {"sol": sol})
        sol.client.close()
    assert orders == mix_orders and [n["played"] for n in notes] == [n["played"] for n in mix_notes]
    choice = [n for n in notes if n["legal"] > 1]
    assert choice and all(n["arm"] == "sol" for n in notes)
    for n in choice:
        lu = n["luna"]
        assert lu["asked"] and lu["choice"] == lu["mix"] == n["played"] and not lu["fell_back"]
        assert n["source"] == "sol" and lu["who"] == "sol" and lu["model"] == "gpt-6-sol"
        assert lu["tokens_from"] == "usage" and lu["output_tokens"] > 0 and lu["cost_usd"] > 0 and lu["reason"].startswith("The fake's reason")
        assert "lines" not in n and "LOOK-AHEAD" not in lu["brief"]
    games = fake.game_requests()
    assert len(games) == len(choice) and all(r["problems"] == [] and r["api"] == "responses" for r in fake.requests)
    for r, n in zip(games, choice):
        system, user = r["body"]["input"]
        assert system["content"].startswith(lp.INSTRUCTIONS) and lp.LINES_NOTE not in system["content"]
        assert user["content"].startswith(n["luna"]["brief"])
        (q,) = r["questions"]
        assert q["choices"] == n["luna"]["choices"]            # the same choices, word for word, as Luna would be asked
    for player, events in zip(players, battles):
        s = player.game_summary(events[0]["room"])
        mine = [n for n in player.notes if n["legal"] > 1]
        assert s["arm"] == "sol" and s["luna_model"] == "gpt-6-sol" and s["luna_answered"] == s["luna_played"] == len(mine)
        assert s["luna_cost_usd"] == pytest.approx(sum(n["luna"]["cost_usd"] for n in mine), abs=1e-5) and s["luna_output_tokens"] > 0
        assert s["budget"]["decide_s"] == 5.0 and s["late"] == 0
        assert "Sol: asked on %d of %d decisions" % (len(mine), len(mine)) in player.search_summary()      # the run's line names the model
    assert FAKE_KEY not in json.dumps(notes)


@tsp.needs_player
def test_sol_gets_its_own_seconds_after_the_search_where_luna_gets_what_the_deadline_leaves(classes):
    battles = tsp.load_fixture()[:1]
    # the search takes 0.45 s of a 0.6 s deadline and the model answers 0.4 s after it is asked
    with FakeDecisions(key=FAKE_KEY, mode="mix", delay=(0.4, 0.41)) as fake:
        sol = sol_for(fake, timeout=2.0)
        budget = sp.SearchBudget(search_ms=20, deadline=0.6, decide_s=2.0, max_decision_s=12.0)
        bridge_s = dict(search_s=0.45)
        notes, _orders, _players = play_bridge(classes, battles, "sol", {"sol": sol}, budgets={"sol": budget}, deadline=0.6, **bridge_s)
        sol.client.close()
        choice = [n for n in notes if n["legal"] > 1 and n["search"] is not None]
        assert choice and all(n["luna"]["asked"] and n["luna"]["choice"] is not None and n["source"] == "sol" for n in choice)
        assert all(0.8 < n["ms"] / 1000.0 < 2.6 for n in choice)                 # the search, then the model: past the search's deadline
        assert all(n["deadline_s"] > 0.6 for n in choice)
        # Luna's arm has no seconds of its own: with 0.15 s left of the deadline it gets no answer in time, and mix plays
        luna = tl.luna_for(fake, timeout=2.0)
        notes, _orders, _players = play_bridge(classes, battles, "luna", {"luna": luna}, deadline=0.6, **bridge_s)
        luna.client.close()
        choice = [n for n in notes if n["legal"] > 1 and n["search"] is not None]
        assert choice and all(n["luna"]["fell_back"] and n["luna"]["why"] in ("timeout", "no_time") for n in choice)
        assert all(n["ms"] < 800 for n in choice)


def play_bridge(classes, battles, mode, deciders, search_s=0.0, deadline=1.5, **kw):
    bridge = tsp.FakeBridge(battles, search_s=search_s)
    notes, orders, players = [], [], []
    try:
        for events in battles:
            mine = []
            player = tsp.searcher(classes, events[-1]["username"], bridge, mode=mode, deadline=deadline, on_decision=mine.append,
                                  deciders=deciders, **kw)
            tsp.play(player, events)
            orders.append(tsp.chosen(player))
            players.append(player)
            player.notes = mine
            notes += mine
    finally:
        bridge.close()
    players.append(bridge)
    return notes, orders, players


@tsp.needs_player
def test_mix_lines_asks_for_lines_and_plays_the_clearly_better_one(classes):
    plain = tsp.load_fixture()[:2]
    mix_notes, mix_orders, _ = tl.play_arm(classes, plain, "mix")
    # lines that all end alike: the arm plays what mix plays, and every searched decision carries its lines
    level = with_lines(plain, values=(0.50, 0.52, 0.47))          # no line 0.10 better than another: nothing changes
    notes, orders, players = play_bridge(classes, level, "mix-lines", {})
    bridge = players.pop()
    assert orders == mix_orders
    searched = [n for n in notes if n["search"] is not None]
    assert searched and all("lines" in n for n in searched) and not any("luna" in n for n in notes)
    assert all(n["lines"]["changed"] is False and n["lines"]["ms"] == 1234.5 for n in searched)
    assert any(len(n["lines"]["values"]) >= 2 for n in searched)
    assert all(more.get("lines") == {"k": 3, "turns": 5, "ms": 1500.0, "worlds": 2, "threads": 4} for (_k, _r, _ttl, _ms, more) in bridge.asked)
    told = [n["lines"]["text"] for n in searched if n["lines"].get("text")]
    assert told and all("after 5 turns" in t and "*" in t for t in told)
    for player, events in zip(players, level):
        s = player.game_summary(events[0]["room"])
        assert s["arm"] == "mix-lines" and s["lines_changed"] == 0 and s["lines_read"] == sum(1 for n in player.notes if n.get("lines", {}).get("values"))
        assert s["budget"]["lines"] == {"k": 3, "turns": 5, "ms": 1500.0, "worlds": 2, "threads": 4}
    # the second favourite's line ends 0.20 better: where the search rates it and mix played the favourite, it is played
    notes, orders, players = play_bridge(classes, with_lines(plain), "mix-lines", {})
    players.pop()
    changed = [n for n in notes if n.get("lines", {}).get("changed")]
    assert changed and orders != mix_orders
    for n in changed:
        info = n["lines"]
        assert n["source"] == "lines" and n["played"] == info["played"] != info["mix"]
        assert any(info["gain"] == pytest.approx(g) for g in (0.10, 0.20, 0.30))        # third to first, first to second, third to second
        assert n["mask"][n["played"]] == 1 and info["mix_text"] and info["why"] == "line"
    kept = [n for n in notes if "lines" in n and not n["lines"]["changed"] and len(n["lines"]["values"]) >= 2]
    assert all(n["lines"]["why"] in ("mix_best", "close", "no_line_for_mix") for n in kept)
    assert sum(p.game_summary(e[0]["room"])["lines_changed"] for p, e in zip(players, plain)) == len(changed)
    # the plain arms never ask for lines
    notes, _orders, players = play_bridge(classes, level, "mix", {})
    assert all("lines" not in more for (_k, _r, _ttl, _ms, more) in players.pop().asked) and not any("lines" in n for n in notes)


@tsp.needs_player
def test_a_model_with_lines_reads_them_in_its_brief(classes):
    battles = with_lines(tsp.load_fixture()[:1])
    mix_notes, mix_orders, _ = tl.play_arm(classes, tsp.load_fixture()[:1], "mix")
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        sol, luna = sol_for(fake), tl.luna_for(fake)
        for mode, who in (("sol-lines", {"sol": sol}), ("luna-lines", {"luna": luna})):
            before = len(fake.game_requests())
            notes, orders, players = play_bridge(classes, battles, mode, who)
            players.pop()
            assert orders == mix_orders                         # the model answers mix's pick: the lines change nothing by themselves
            choice = [n for n in notes if n["legal"] > 1]
            read = [n for n in choice if n.get("lines", {}).get("values")]
            assert read and all(n["lines"]["changed"] is False and n["lines"]["why"] == "shown" for n in read)
            games = fake.game_requests()[before:]
            assert len(games) == len(choice)
            for r, n in zip(games, choice):
                brief = n["luna"]["brief"]
                has = bool(n.get("lines", {}).get("values"))
                assert ("LOOK-AHEAD LINES" in brief) == has
                if mode == "sol-lines":
                    assert (lp.LINES_NOTE in r["body"]["input"][0]["content"]) == has
                else:
                    assert r["body"]["questions"][0]["instructions"] == lp.INSTRUCTIONS + (lp.LINES_NOTE if has else "")
                if has:
                    assert "Value 0." in brief and "T1 we use Earthquake and foe uses Outrage" in brief
                    assert sum(", line 0." in c["description"] for c in n["luna"]["choices"]) == len(n["lines"]["values"])
                    if mode == "luna-lines":             # (the estimate; Sol's count is the fake API's own)
                        assert n["luna"]["input_tokens"] <= lp.TOKEN_BUDGET + lp.LINES_TOKENS
            s = players[0].game_summary(battles[0][0]["room"])
            assert s["arm"] == mode and s["lines_read"] == len(read) and s["luna_answered"] == len(choice)
        sol.client.close()
        luna.client.close()


@tsp.needs_player
def test_a_model_arm_without_its_model_is_mix_and_says_off(classes):
    battles = tsp.load_fixture()[:1]
    _mix_notes, mix_orders, _ = tl.play_arm(classes, battles, "mix")
    notes, orders, players = play_bridge(classes, battles, "sol", {})
    players.pop()
    assert orders == mix_orders
    choice = [n for n in notes if n["legal"] > 1]
    assert choice and all(n["luna"]["fell_back"] and n["luna"]["why"] == "off" and n["luna"]["who"] == "sol" for n in choice)


# =============================================================================================
# The model alone (luna-solo, sol-solo): no search, the network's pick never played
# =============================================================================================

class _Type:
    def __init__(self, name):
        self.name = name


class _SoloMove:
    def __init__(self, name, power, kind="PHYSICAL", accuracy=1.0, type_=None):
        self.id, self.base_power, self.accuracy, self.type = name, power, accuracy, type_
        self.category = _Type(kind)


class _SoloMon:
    def __init__(self, hp=1.0, types=(), weak=None):
        self.current_hp_fraction, self.types, self._weak = hp, types, dict(weak or {})

    def damage_multiplier(self, move):
        return self._weak.get(move.id, 1.0)


def test_the_fallback_rule_of_a_model_alone_is_the_strongest_attack_or_the_healthiest_switch():
    fire, water = _Type("FIRE"), _Type("WATER")
    battle = type("B", (), {"active_pokemon": _SoloMon(types=(fire,)), "opponent_active_pokemon": _SoloMon(weak={"surf": 2.0, "tackle": 0.0})})()
    moves = {0: (_SoloMove("swordsdance", 0, "STATUS"), {}), 1: (_SoloMove("flamethrower", 90, type_=fire), {}),
             2: (_SoloMove("surf", 80, type_=water), {}), 3: (_SoloMove("tackle", 250), {}),
             11: (_SoloMove("flamethrower", 90, type_=fire), {"terastallize": True}), 5: (_SoloMon(hp=0.9), {})}
    mask = [1.0 if a in moves else 0.0 for a in range(14)]
    assert lp.solo_rule(battle, mask, moves) == 2                      # Surf: 80 x 2 beats Flamethrower's 90 x 1.5, and Tackle does nothing
    battle.opponent_active_pokemon = _SoloMon()
    assert lp.solo_rule(battle, mask, moves) == 3                      # nothing is weak or immune: the largest power
    moves[3] = (_SoloMove("tackle", 250, accuracy=0.3), {})
    assert lp.solo_rule(battle, mask, moves) == 1                      # 75 a go: the sure, same-type move is worth more (never its Tera form)
    only_status = {0: (_SoloMove("swordsdance", 0, "STATUS"), {}), 1: (_SoloMove("recover", 0, "STATUS"), {}), 6: (_SoloMon(hp=1.0), {})}
    assert lp.solo_rule(battle, [1.0 if a in only_status else 0.0 for a in range(14)], only_status) == 0   # no attack: the first move, not a switch
    forced = {4: (_SoloMon(hp=0.3), {}), 6: (_SoloMon(hp=0.8), {}), 7: (_SoloMon(hp=0.8), {}), 9: (_SoloMon(hp=0.1), {})}
    assert lp.solo_rule(battle, [1.0 if a in forced else 0.0 for a in range(14)], forced) == 6          # the healthiest, the first of two alike
    illegal = dict(moves)
    assert lp.solo_rule(battle, [1.0 if a == 0 else 0.0 for a in range(14)], illegal) == 0             # only what the request allows
    with pytest.raises(ValueError):
        lp.solo_rule(battle, [0.0] * 14, moves)


def test_a_model_alone_is_given_its_own_last_moves_and_the_turns_before():
    log = lp.TurnLog(keep=3)
    for turn in range(1, 6):
        log.feed([["", "turn", str(turn)], ["", "move", "p1a: Garchomp", "Earthquake", "p2a: Heatran"], ["", "-damage", "p2a: Heatran", "%d/100" % (90 - turn)]])
    assert log.number == 5 and len(log.turn) == 2 and log.last[1][3] == "86/100"      # turn 5 is being played, turn 4 is the last one
    assert [t for t, _lines in log.history] == [2, 3, 4] and [t for t, _lines in log.earlier()] == [2, 3]
    log.feed([["", "turn", "6"]])
    assert [t for t, _lines in log.history] == [3, 4, 5] and log.last is log.history[-1][1]
    assert [t for t, _lines in log.earlier()] == [3, 4]                        # the last turn is told on its own ("LAST TURN")
    facts = tl.toy_facts(3)
    facts.solo = True
    facts.memory = [
        {"turn": 20, "text": "switch Klefki", "reason": "Klefki takes the hit.", "switch": True, "forced": False, "by": "model"},
        {"turn": 21, "text": "Play Rough", "reason": None, "switch": False, "forced": False, "by": "rule"},
        {"turn": 21, "text": "switch Luvdisc", "reason": "It outspeeds.", "switch": True, "forced": True, "by": "model"},
        {"turn": 22, "text": "switch Emboar", "reason": "Emboar threatens a knockout.", "switch": True, "forced": False, "by": "model"},
    ]
    board = type("Board", (), {"facts": facts})()
    text = lp._memory(board, 0)
    assert text.startswith("YOUR LAST MOVES (oldest first): T20 switch Klefki [your reason: Klefki takes the hit.]; "
                           "T21 Play Rough (no answer from you: a fallback rule chose); T21 switch Luvdisc (forced) [your reason: It outspeeds.]; "
                           "T22 switch Emboar [your reason: Emboar threatens a knockout.].")
    assert "Of your last 3 free moves, 2 were switches. That is a lot" in text
    facts.memory = facts.memory[1:2]
    assert lp._memory(board, 0).endswith("Of your last 1 free moves, 0 were switches.")
    assert "your reason" not in lp._memory(type("Board", (), {"facts": tl.toy_facts(3)})(), 0) and lp._memory(type("Board", (), {"facts": tl.toy_facts(3)})(), 0) == ""
    for words in ("SWITCHING IS EXPENSIVE", "YOUR LAST MOVES", "FOE'S POSSIBLE MOVES", "do not assume an immunity"):
        assert words in lp.SOLO_INSTRUCTIONS


def test_the_solo_arms_are_modes_that_start_no_search():
    for arm, model in (("luna-solo", "luna"), ("sol-solo", "sol")):
        assert arm in sp.MODES and arm in ladder.PLAYER_MODES and sp.split_mode(arm) == ("net", None) and sp.DECIDERS[arm] == model
        args = ladder.parse_args(["--player", arm, "--helpers", "1", "--parallel", "3"])
        assert ladder.decider_models(args) == [model] and ladder.wants_luna(args)
        assert ladder.search_problems(args) == [] and ladder.arm_budgets(args, [arm]) == {}      # no helpers needed, no search budget
    assert sp.SOLO_ARMS == ("luna-solo", "sol-solo") + sp.CLAUDE_SOLO_ARMS
    assert "no adviser is behind you" in lp.SOLO_INSTRUCTIONS and "net =" not in lp.SOLO_INSTRUCTIONS and "MIX" not in lp.SOLO_INSTRUCTIONS


@tsp.needs_player
def test_a_model_alone_plays_its_own_moves_and_sees_no_advisers(classes):
    battles = tsp.load_fixture()[:2]
    with FakeDecisions(key=FAKE_KEY, mode="last") as fake:
        luna = tl.luna_for(fake)
        notes, orders, players = play_bridge(classes, battles, "luna-solo", {"luna": luna})
        bridge = players.pop()
        luna.client.close()
    assert bridge.asked == []                                          # the search was never asked
    choice = [n for n in notes if n["legal"] > 1]
    assert choice and all(n["arm"] == "luna-solo" and n["search"] is None for n in notes)
    games = fake.game_requests()
    assert len(games) == len(choice) and all(r["problems"] == [] for r in fake.requests)
    for r, n in zip(games, choice):
        lu = n["luna"]
        (q,) = r["body"]["questions"]
        assert q["instructions"] == lp.SOLO_INSTRUCTIONS
        ids = sorted(int(c["value"].split("-")[0][1:]) for c in q["choices"])
        assert ids == [a for a in range(14) if n["mask"][a]]              # every legal action is on offer, and nothing else
        text = r["body"]["input"] + " ".join(c["description"] for c in q["choices"])
        for word in ("MIX'S PICK", "ADVISORS", "The network alone", "network's", "win estimate", "mix would", "mix's"):
            assert word not in text, (word, text[:300])
        assert not re.search(r"\b(net|search) (<1|\d+)%", text)            # no adviser's number on any choice
        assert not any(c["description"] != c["description"].rstrip() for c in q["choices"])
        # what a model alone is told instead: what the foe can carry, and what a switch costs
        assert "FOE'S POSSIBLE MOVES (" in r["body"]["input"] and "shown: " in r["body"]["input"]
        switches = [c for c in q["choices"] if "-switch-" in c["value"]]
        assert all(("COST: it uses our turn" in c["description"]) == (not n["forced"]) for c in switches)
        # the model's choice is the order: the last choice offered, whatever the network or the rule would have played
        assert lu["solo"] and lu["asked"] and not lu["fell_back"] and not lu["overruled"] and n["source"] == "luna"
        assert n["played"] == lu["choice"] == int(q["choices"][-1]["value"].split("-")[0][1:])
        assert n["mask"][lu["mix"]] == 1                                  # ("mix" here: the fallback rule's pick, a legal action)
    assert any(n["played"] != n["net"] for n in choice)                   # the network's pick is written down, and is not what plays
    # its memory: from its second move of a battle on, the brief says what it did before, and counts its switches
    for player in players:
        mine = [n for n in player.notes if n["legal"] > 1]
        assert "YOUR LAST MOVES" not in mine[0]["luna"]["brief"]
        for before, n in zip(mine, mine[1:]):
            brief = n["luna"]["brief"]
            assert "YOUR LAST MOVES (oldest first): " in brief and ("T%s %s" % (before["turn"], before["luna"]["choice_text"])) in brief
            assert re.search(r"Of your last \d free moves, \d were switches\.", brief) or "(forced)" in brief
        assert any("EARLIER TURNS: T" in n["luna"]["brief"] for n in mine[3:])
        assert all(n["luna"]["input_tokens"] <= lp.TOKEN_BUDGET + lp.SOLO_TOKENS for n in mine)
    for player, events in zip(players, battles):
        s = player.game_summary(events[0]["room"])
        mine = [n for n in player.notes if n["legal"] > 1]
        assert s["arm"] == "luna-solo" and s["luna_solo"] is True and s["luna_played"] == len(mine) and s["searched"] == 0
        assert "Luna alone: asked on %d of %d decisions with a choice, answered and played %d" % (len(mine), len(mine), len(mine)) in player.search_summary()
    assert FAKE_KEY not in json.dumps(notes)


@tsp.needs_player
@pytest.mark.parametrize("behaviour, why", [("http_500", "http_500"), ("hang", "timeout"), ("illegal", "illegal")])
def test_a_model_alone_that_gives_no_move_gets_the_rule_not_the_network(classes, behaviour, why):
    battles = tsp.load_fixture()[:1]
    with FakeDecisions(key=FAKE_KEY, mode=behaviour, hang_s=1.0) as fake:
        sol = sol_for(fake, timeout=0.4)
        sol.client.breaker_failures = 10 ** 6
        notes, _orders, players = play_bridge(classes, battles, "sol-solo", {"sol": sol})
        bridge = players.pop()
        sol.client.close()
    assert bridge.asked == []
    choice = [n for n in notes if n["legal"] > 1]
    assert choice and all(n["luna"]["fell_back"] and n["luna"]["why"] == why and n["luna"]["solo"] for n in choice)
    assert all(n["source"] == "rule" and n["played"] == n["luna"]["mix"] and n["mask"][n["played"]] == 1 for n in choice)
    assert all(n["played"] < 10 for n in choice)                          # the rule never Terastallizes
    assert any(n["played"] != n["net"] for n in choice)                   # and it is not the network's pick under another name
    assert all(n["ms"] < 1500 for n in choice)                            # no search to wait for: the model's own timeout and no more
    s = players[0].game_summary(battles[0][0]["room"])
    assert s["luna_played"] == 0 and s["luna_fallbacks"] == {why: len(choice)}
    assert "the fallback rule played instead on %d" % len(choice) in players[0].search_summary()
    # without a key at all: the rule plays every move, and says "off"
    notes, _orders, players = play_bridge(classes, battles, "luna-solo", {})
    players.pop()
    choice = [n for n in notes if n["legal"] > 1]
    assert choice and all(n["luna"]["why"] == "off" and n["source"] == "rule" and n["played"] == n["luna"]["mix"] for n in choice)


def test_the_report_and_the_launcher_know_the_arms(tmp_path):
    path = tmp_path / "results.jsonl"
    rows_ = [{"arm": "sol", "won": True, "luna_decisions": 30, "luna_asked": 30, "luna_answered": 29, "luna_played": 29, "luna_agreed_with_mix": 20,
              "luna_changed": 9, "luna_tokens": 60000, "luna_output_tokens": 1800, "luna_cost_usd": 0.138, "luna_fallbacks": {"timeout": 1},
              "luna_latency_ms_mean": 1500.0, "luna_latency_ms_max": 4100.0, "luna_model": "gpt-6-sol"},
             {"arm": "luna", "won": False, "luna_decisions": 30, "luna_asked": 30, "luna_answered": 30, "luna_played": 30,
              "luna_agreed_with_mix": 28, "luna_changed": 2, "luna_tokens": 60000, "luna_cost_usd": 0.006, "luna_fallbacks": {}}]
    path.write_text("\n".join(json.dumps(r) for r in rows_) + "\n", encoding="utf-8")
    sol = luna_report.summary(luna_report.load(str(path), arm="sol"), "Sol")
    assert sol["games"] == 1 and sol["wins"] == 1 and sol["luna_cost_usd"] == pytest.approx(0.138) and sol["text"].startswith("Sol: asked on 30 of 30")
    assert luna_report.text(sol, None, "sol").startswith("Sol: 1 games, 1 wins and 0 losses")
    luna = luna_report.summary(luna_report.load(str(path)))
    assert luna["games"] == 1 and luna["luna_cost_usd"] == pytest.approx(0.006) and luna_report.text(luna, None).startswith("Luna: 1 games")
    assert luna_report.text(luna_report.summary([]), None, "sol-lines") == "no games of the arm sol-lines yet"
    assert ", Sol: asked on 30 of 30" in ladder.luna_text(rows_[:1]) and ", Luna: asked on 30 of 30" in ladder.luna_text(rows_[1:])
    assert ", the model: asked on 60 of 60" in ladder.luna_text(rows_)
    text = open(tl.LAUNCHER, encoding="utf-8").read()
    assert "--luna-check --model %CHECKMODEL%" in text and "--arms mix,%ARM% --experiment %ARM%-vs-mix-024381" in text
    assert "--player %ARM% " in text and "luna_report.py --arm %ARM%" in text and "set ARM=%ARM%-lines" in text
    assert "\r\n" in open(tl.LAUNCHER, encoding="utf-8", newline="").read()


def test_the_window_app_offers_the_models_alone_and_never_stores_the_key():
    source = open(os.path.join(REPO, "app", "PSBotLookahead.cs"), encoding="utf-8-sig").read()
    # (the modes where a model had the last word over the bot, and reading ahead, left the window on 2026-10-14;
    #  the arms themselves are still ladder.py's)
    for text in ('"Bot: network + search (Lookahead)", "Bot: network only",',
                 '"GPT-6 Sol alone (OpenAI)", "GPT-6 Luna alone (OpenAI)" };',
                 'static readonly string[] ModeKeys = { "lookahead", "net", "fable", "opus", "sonnet", "haiku", "solalone", "lunaalone" };',
                 'if (mode == "lunaalone") return "luna-solo";', 'if (mode == "solalone") return "sol-solo";',
                 'if (saved == "lunasolo") return "lunaalone";', 'if (saved == "luna" || saved == "sol") return "lookahead";',
                 'bool UsesSearch { get { return ModeKey == "lookahead"; } }',                   # a model alone starts no search helpers
                 'args.Add("--player"); args.Add(PlayerOf(mode, false));', 'MessageBoxDefaultButton.Button2) != DialogResult.Yes) return;',
                 'with no network and no search behind it.',
                 'if (model) args.Add("--key-stdin");', 'keyBox.UseSystemPasswordChar = true;', 'keyBox.Text = "";'):
        assert text in source, text
    assert 'aheadBox' not in source and '-lines"' not in source
    # the key is written to the child's standard input after the password's line, and is in no argument and no settings line
    assert 'GetBytes(passBox.Text + "\\n" + (model ? keyBox.Text.Trim() + "\\n" : ""))' in source
    assert 'kv.Key != "password" && kv.Key != "key" && kv.Key != "openaikey"' in source
    assert "settings[\"openaikey\"]" not in source and "settings[\"key\"]" not in source
    assert "args.Add(keyBox" not in source and "EnvironmentVariables[\"OPENAI_API_KEY\"]" not in source
    assert "EnvironmentVariables[\"ANTHROPIC_API_KEY\"]" not in source and "settings[\"apikey\"]" not in source


def _real_foul_play() -> bool:
    if os.name == "nt" and not os.environ.get("PSBOT_TEST_WSL"):
        return False                 # (on Windows the probe starts the WSL machine: only when asked for)
    try:
        return bool(fb.available()[0]) and bool(fb.load_trace())
    except Exception:
        return False


needs_foul_play = pytest.mark.skipif(not _real_foul_play(), reason="needs Foul Play and poke-engine (the real search helper)")


def _named(fp_name: str) -> str:
    """The helper's words for one of its own option names (fp_bridge_server._opponent_name)."""
    if fp_name.startswith("switch "):
        return fp_name
    if fp_name in ("No Move", "none", ""):
        return "none"
    for suffix, word in (("-tera", " terastallize"), ("-mega", " mega")):
        if fp_name.endswith(suffix):
            return "move " + fp_name[: -len(suffix)] + word
    return "move " + fp_name


@needs_foul_play
def test_the_real_helper_plays_its_favourites_forward():
    """Foul Play's own engine: the lines of a whole recorded battle, every one checked for its shape and its sense."""
    events = fb.load_trace()[0]
    user, room = events[-1]["username"], events[0]["room"]
    key = user + "|" + room
    spec = {"k": 3, "turns": 5, "ms": 900, "worlds": 2, "threads": 2}
    bridge = fb.FoulPlayBridge(helpers=1, search_ms=100, timeout=120.0).start()
    seen = with_lines_count = full = forced_steps = ended = 0
    took = []
    try:
        for event in events:
            if event["t"] == "msg":
                bridge.feed(key, user, room, event["data"])
            elif event["t"] == "order":
                seen += 1
                advice = bridge.advice_sync(key, rqid=event["rqid"], lines=spec)
                bridge.note(key, event["choice"])
                plain = advice.lines
                assert isinstance(plain, dict) and "error" not in plain, plain
                assert (plain["k"], plain["turns"], plain["budget_ms"], plain["threads"]) == (3, 5, 900.0, 2) and plain["step_ms"] >= 2.0
                took.append(plain["ms"])
                offered = {o.choice for o in advice.options}
                assert 1 <= len(plain["candidates"]) <= 3 and set(plain["candidates"]) <= offered
                assert 1 <= len(plain["worlds"]) <= 2
                chances = [w["chance"] for w in plain["worlds"]]
                assert chances == sorted(chances, reverse=True)                    # the likeliest worlds, the likeliest first
                for world in plain["worlds"]:
                    assert "error" not in world, world
                    for one in world["lines"]:
                        assert "error" not in one, one
                        with_lines_count += 1
                        assert one["choice"] in plain["candidates"] and one["steps"], one
                        assert one["value"] is None or 0.0 <= one["value"] <= 1.0
                        assert one["end"] is None or 0.0 <= one["end"] <= 1.0
                        assert one["turns"] == sum(1 for s in one["steps"] if not s.get("forced")) <= 5
                        assert one["steps"][0]["ours"] == _named(one["fp"])            # the line starts with the action it is the line of
                        for s in one["steps"]:
                            forced_steps += int(bool(s.get("forced")))
                            assert 0.0 < s["p"] <= 1.0 and set(s["after"]) == {"ours", "theirs"}
                            for side in s["after"].values():
                                assert 0 <= side["hp"] <= 100 and 0 <= side["left"] <= 6 and 0.0 <= side["team_hp"] <= 1.0 and side["active"]
                        last = one["steps"][-1]["after"]
                        over = last["ours"]["left"] == 0 or last["theirs"]["left"] == 0
                        ended += int(over)
                        if over:
                            assert one["end"] in (0.0, 1.0)
                        full += int(one["turns"] == 5)
        assert seen >= 10 and with_lines_count >= 2 * seen
        assert full + ended >= 0.7 * with_lines_count                                      # most lines reach their five turns or the game's end
        assert bridge.counters["decision_errors"] == 0
        assert max(took) < 2.0 * 900 + 400 and sum(took) / len(took) < 1.25 * 900 + 150   # the budget holds
        stats = bridge.stats()
        assert stats.get("decisions", seen) >= seen
    finally:
        bridge.close()
    print("lines: %d decisions, %d lines (%d of five turns, %d ended the game, %d forced steps), %.0f ms on average, %.0f at the slowest"
          % (seen, with_lines_count, full, ended, forced_steps, sum(took) / len(took), max(took)))
