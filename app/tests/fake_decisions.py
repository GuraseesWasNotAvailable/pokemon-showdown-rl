"""
A fake of OpenAI's Decisions API, for the tests and the smoke games of the Luna arm (luna_player.py).
No key of OpenAI's is involved anywhere: the fake has a key of its own that it expects.

    from fake_decisions import FakeDecisions
    with FakeDecisions(key="sk-fake-...", mode="mix") as fake:
        client = LunaClient("sk-fake-...", url=fake.url)
        ...
        fake.requests            every request as it came, checked against the documented shape

    FAKE_LUNA_KEY=sk-fake-... python tests/fake_decisions.py --port 11450 --mode random \\
        --delay 0.05 0.9 --error-rate 0.2 --log runs/fake_decisions.jsonl

The request is held to the documented shape (https://developers.openai.com/api/docs/guides/decisions):
POST /v1/decisions, Authorization: Bearer KEY, Content-Type: application/json, a JSON object with
exactly "model" ("gpt-6-luna"), "input" (a text, or user messages of input_text and input_image
parts) and "questions" (each "predicate", "choice" with "choices" of value and description, or
"score" with "levels" of label and description; names unique). Anything else is answered as the
real API answers a bad request: HTTP 400 (401 for the key, 404 for the model) with
{"error": {"message", "type", "param", "code"}}; `problems` in the recorded request says what was wrong.

The answer is the documented one: {"answers": [{"type": "choice", "name", "choice",
"probabilities": [{"value", "probability"}], "confidence"}]} (a predicate: "probability"; a score:
"score", "probabilities" with "value" and "label", "confidence").

How a choice question is answered (mode, or for the next requests one by one: script):

    mix            the choice whose description says MIX'S PICK (the first when none does)
    random         a random one of the choices
    first, last    that one
    other          the first choice that is not mix's pick
    worst          the choice with the least of "net N%" plus "search N%" in its description (a blunder)
    value:TEXT     that value, offered or not
    illegal        a value that was not offered
    refusal        an answer of type "refusal"
    malformed_json     HTTP 200 with a body that is not JSON
    malformed_shape    HTTP 200 with JSON that has no "answers" list
    no_choice          a choice answer without its "choice"
    wrong_name         an answer under another name
    http_401 ... http_503    that HTTP error, in the API's error shape (401 echoes the key it was given, as a test of redaction)
    html           HTTP 502 with a web page (a proxy's)
    reset          the connection closed with no answer
    hang           no answer for hang_s seconds (the client's timeout comes first)
    delay:S        the answer of `mode`, S seconds late

delay=(low, high) adds a random wait to every answer; error_rate is the share of the game's
questions answered with one of `errors` instead (drawn at random). switch_after=N with
switch_mode=M answers every game question after the first N with M (a key that stops working in
the middle of a run: --switch-after 30 --switch-mode http_401). The check question of
ladder.py --luna-check (the choice named "sum") is always answered "four", and the script, the
errors and the delays leave it alone, unless fail_checks is set; so they do the check's question
of a game's size (the choice named "sample").

The same server also fakes the Responses API for the Sol arms (POST /v1/responses, .responses_url,
model "gpt-6-sol"): the request is held to the shape luna_player.responses_body writes (a system
and a user message, text.format a strict JSON schema with one object per question, reasoning.effort,
max_output_tokens, store) and to the strict-schema rules (additionalProperties false, every
property required), and answered in the Responses shape: {"status": "completed", "output":
[{"type": "message", "content": [{"type": "output_text", "text": "<the JSON object>"}]}],
"usage": {...}}. The behaviours above hold for it too (a refusal is a content part of type
"refusal"), and two more: "incomplete" (status incomplete, no output text) and refuse_efforts,
the reasoning efforts answered with HTTP 400 as the real API answers one a model does not take.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

MODEL = "gpt-6-luna"
PATH = "/v1/decisions"
SOL_MODEL = "gpt-6-sol"
RESPONSES_PATH = "/v1/responses"
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
MIX_MARK = "MIX'S PICK"
DEFAULT_ERRORS = ("http_500", "http_429", "http_503", "hang", "refusal", "malformed_json", "malformed_shape", "illegal", "reset")
_ERRORS = {
    400: ("invalid_request_error", None, "The request is not valid."),
    401: ("invalid_request_error", "invalid_api_key", "Incorrect API key provided: %s. You can find your API key at https://platform.openai.com/account/api-keys."),
    403: ("invalid_request_error", "unsupported_country_region_territory", "Country, region, or territory not supported"),
    404: ("invalid_request_error", "model_not_found", "The model `gpt-6-luna` does not exist or you do not have access to it."),
    429: ("rate_limit_error", "rate_limit_exceeded", "Rate limit reached for gpt-6-luna on requests per min. Please try again in 1s."),
    500: ("server_error", None, "The server had an error while processing your request. Sorry about that!"),
    503: ("server_error", None, "The engine is currently overloaded, please try again later."),
}


def check_request(body: Any) -> List[str]:
    """What is wrong with a request body against the documented shape ([] when nothing is)."""
    problems: List[str] = []
    if not isinstance(body, dict):
        return ["the body is not a JSON object"]
    extra = sorted(set(body) - {"model", "input", "questions"})
    if extra:
        problems.append("unknown fields: %s" % ", ".join(extra))
    for name in ("model", "input", "questions"):
        if name not in body:
            problems.append("missing %s" % name)
    source = body.get("input")
    if isinstance(source, str):
        if not source.strip():
            problems.append("input is an empty text")
    elif isinstance(source, list) and source:
        for message in source:
            if not isinstance(message, dict) or message.get("role") != "user" or not isinstance(message.get("content"), list):
                problems.append("an input message is not {role: user, content: [...]}")
                continue
            for part in message["content"]:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind == "input_text":
                    if not isinstance(part.get("text"), str):
                        problems.append("an input_text part without text")
                elif kind == "input_image":
                    if not str(part.get("image_url") or "").startswith("data:image/"):
                        problems.append("an input_image part that is not an inline base64 data URL")
                else:
                    problems.append("an input part of type %r" % (kind,))
    elif "input" in body:
        problems.append("input is neither a text nor a list of user messages")
    questions = body.get("questions")
    if "questions" in body and (not isinstance(questions, list) or not questions):
        problems.append("questions is not a list with at least one question")
        questions = []
    names: List[str] = []
    for q in questions or []:
        if not isinstance(q, dict):
            problems.append("a question is not an object")
            continue
        kind = q.get("type")
        name = q.get("name")
        if not isinstance(name, str) or not name:
            problems.append("a question without a name")
        elif name in names:
            problems.append("two questions named %s" % name)
        else:
            names.append(name)
        if not isinstance(q.get("instructions"), str) or not q.get("instructions"):
            problems.append("question %s: no instructions" % name)
        allowed = {"type", "name", "instructions"}
        if kind == "predicate":
            pass
        elif kind == "choice":
            allowed.add("choices")
            choices = q.get("choices")
            if not isinstance(choices, list) or len(choices) < 2:
                problems.append("question %s: choices is not a list of two or more" % name)
            else:
                values = []
                for c in choices:
                    if not isinstance(c, dict) or set(c) != {"value", "description"}:
                        problems.append("question %s: a choice is not exactly {value, description}" % name)
                        continue
                    if not isinstance(c["value"], str) or not c["value"]:
                        problems.append("question %s: a choice's value is not a text" % name)
                    if not isinstance(c["description"], str) or not c["description"]:
                        problems.append("question %s: a choice without a description" % name)
                    values.append(c.get("value"))
                if len(set(values)) != len(values):
                    problems.append("question %s: a value is offered twice" % name)
        elif kind == "score":
            allowed.add("levels")
            levels = q.get("levels")
            if not isinstance(levels, list) or len(levels) < 2 or any(
                    not isinstance(lv, dict) or set(lv) != {"label", "description"} for lv in levels):
                problems.append("question %s: levels is not a list of {label, description}" % name)
        else:
            problems.append("question %s: type %r is not predicate, choice or score" % (name, kind))
        more = sorted(set(q) - allowed)
        if more:
            problems.append("question %s: unknown fields %s" % (name, ", ".join(more)))
    return problems


def _strict_object(schema: Any, where: str, problems: List[str]) -> None:
    """The strict-mode rules of structured outputs for one object schema: no extra properties, every property required."""
    if not isinstance(schema, dict) or schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        problems.append("%s is not an object schema with properties" % where)
        return
    if schema.get("additionalProperties") is not False:
        problems.append("%s: additionalProperties is not false" % where)
    if sorted(schema.get("required") or []) != sorted(schema["properties"]):
        problems.append("%s: required is not every property" % where)


def check_responses_request(body: Any) -> List[str]:
    """What is wrong with a Responses API request as luna_player.responses_body writes it ([] when nothing is)."""
    problems: List[str] = []
    if not isinstance(body, dict):
        return ["the body is not a JSON object"]
    extra = sorted(set(body) - {"model", "input", "text", "reasoning", "max_output_tokens", "store"})
    if extra:
        problems.append("unknown fields: %s" % ", ".join(extra))
    for name in ("model", "input", "text"):
        if name not in body:
            problems.append("missing %s" % name)
    messages = body.get("input")
    if not isinstance(messages, list) or not messages:
        problems.append("input is not a list of messages")
    else:
        roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
        if roles.count("user") != 1 or any(r not in ("system", "developer", "user") for r in roles):
            problems.append("input is not system messages and exactly one user message")
        if any(not isinstance(m, dict) or not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            problems.append("an input message without a text")
    fmt = (body.get("text") or {}).get("format") if isinstance(body.get("text"), dict) else None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema" or not isinstance(fmt.get("name"), str) or fmt.get("strict") is not True:
        problems.append("text.format is not a strict json_schema with a name")
    else:
        schema = fmt.get("schema")
        _strict_object(schema, "the schema", problems)
        for name, prop in ((schema or {}).get("properties") or {}).items() if isinstance(schema, dict) else []:
            _strict_object(prop, "the schema's %s" % name, problems)
            choice = (prop.get("properties") or {}).get("choice") if isinstance(prop, dict) else None
            if choice is not None and (not isinstance(choice.get("enum"), list) or len(choice["enum"]) < 2
                                       or len(set(choice["enum"])) != len(choice["enum"])):
                problems.append("the schema's %s: choice is not an enum of two or more different values" % name)
    if "reasoning" in body and (not isinstance(body["reasoning"], dict) or body["reasoning"].get("effort") not in EFFORTS):
        problems.append("reasoning.effort is not one of %s" % ", ".join(EFFORTS))
    tokens = body.get("max_output_tokens")
    if "max_output_tokens" in body and (not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 16):
        problems.append("max_output_tokens is not a number of 16 or more")
    if "store" in body and not isinstance(body["store"], bool):
        problems.append("store is not true or false")
    return problems


def questions_of_responses(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The questions a Responses request asks, in the Decisions shape: from its schema, with each choice's description read back from the user's message."""
    user = next((m.get("content") for m in body.get("input") or [] if isinstance(m, dict) and m.get("role") == "user"), "") or ""
    described: Dict[str, str] = {}
    for line in str(user).splitlines():
        if line.startswith("- ") and ": " in line:
            value, text = line[2:].split(": ", 1)
            described.setdefault(value, text)
    out: List[Dict[str, Any]] = []
    schema = ((body.get("text") or {}).get("format") or {}).get("schema") or {}
    for name, prop in (schema.get("properties") or {}).items():
        choice = (prop.get("properties") or {}).get("choice")
        if choice is not None:
            out.append({"type": "choice", "name": name, "instructions": "",
                        "choices": [{"value": v, "description": described.get(v, "")} for v in choice.get("enum") or []]})
        else:
            out.append({"type": "predicate", "name": name, "instructions": ""})
    return out


class FakeDecisions:
    """The fake server: start() it (or use it in a with block), point the client at .url, read .requests afterwards."""

    def __init__(self, key: str = "sk-fake-luna-0000", port: int = 0, mode: str = "mix", delay: Tuple[float, float] = (0.0, 0.0),
                 error_rate: float = 0.0, errors: Sequence[str] = DEFAULT_ERRORS, seed: Optional[int] = 0, usage: bool = False,
                 hang_s: float = 5.0, log_path: Optional[str] = None, fail_checks: bool = False, host: str = "127.0.0.1",
                 switch_after: Optional[int] = None, switch_mode: str = "http_401", refuse_efforts: Sequence[str] = (),
                 reasoning_s: float = 0.0):
        self.refuse_efforts = tuple(refuse_efforts)     # Responses API: these reasoning efforts are answered with HTTP 400
        self.reasoning_s = float(reasoning_s)           # Responses API: an answer with reasoning (effort not "none") takes this much longer
        self.switch_after = switch_after
        self.switch_mode = switch_mode
        self._game_questions = 0
        self.key = key
        self.mode = mode
        self.delay = (float(delay[0]), float(delay[1]))
        self.error_rate = float(error_rate)
        self.errors = tuple(errors)
        self.usage = bool(usage)
        self.hang_s = float(hang_s)
        self.fail_checks = bool(fail_checks)
        self.log_path = log_path
        self.script: Deque[str] = deque()          # behaviours for the next game questions, one each, before `mode`
        self.requests: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {}
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            wbufsize = 1 << 16                # the headers and the body in one segment (flushed in _send)

            def log_message(self, *args: Any) -> None:      # nothing on stderr
                pass

            def do_GET(self) -> None:
                self._send(405, {"error": {"message": "Method not allowed", "type": "invalid_request_error", "param": None, "code": None}})

            def do_POST(self) -> None:
                fake._handle(self)

            def _send(self, status: int, payload: Any, raw: Optional[bytes] = None, content_type: str = "application/json") -> None:
                data = raw if raw is not None else json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("x-request-id", "req_fake_%06d" % len(fake.requests))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                except Exception:
                    pass

        self._server = ThreadingHTTPServer((host, int(port)), Handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self.url = "http://%s:%d%s" % (host, self.port, PATH)
        self.responses_url = "http://%s:%d%s" % (host, self.port, RESPONSES_PATH)
        self._thread: Optional[threading.Thread] = None

    # -- running it ----------------------------------------------------------------------------------

    def start(self) -> "FakeDecisions":
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, name="fake-decisions", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass

    def __enter__(self) -> "FakeDecisions":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- one request ---------------------------------------------------------------------------------

    @staticmethod
    def _error(status: int, message: Optional[str] = None, fill: str = "") -> Dict[str, Any]:
        kind, code, text = _ERRORS.get(status, ("server_error", None, "error"))
        text = message or (text % fill if "%s" in text else text)
        return {"error": {"message": text, "type": kind, "param": None, "code": code}}

    def _behaviour(self, check: bool) -> str:
        with self._lock:
            if check and not self.fail_checks:
                return "mix"
            self._game_questions += 1
            if self.switch_after is not None and self._game_questions > int(self.switch_after):
                return self.switch_mode
            if self.script:
                return self.script.popleft()
            if self.error_rate > 0 and self._rng.random() < self.error_rate:
                return self._rng.choice(self.errors)
            return self.mode

    def _pick(self, behaviour: str, question: Dict[str, Any]) -> Optional[str]:
        values = [c["value"] for c in question.get("choices") or []]
        marked = [c["value"] for c in question.get("choices") or [] if MIX_MARK in str(c.get("description") or "")]
        if set(values) == {"three", "four", "five"} and not self.fail_checks:
            return "four"                       # the check's question
        if behaviour.startswith("value:"):
            return behaviour[len("value:"):]
        if behaviour == "illegal":
            return "zz-not-offered"
        if behaviour == "random":
            with self._lock:
                return self._rng.choice(values)
        if behaviour == "last":
            return values[-1]
        if behaviour == "other":
            return next((v for v in values if v not in marked), values[0])
        if behaviour == "worst":
            import re

            def support(c: Dict[str, Any]) -> float:
                found = re.findall(r"(?:net|search) (<1|\d+)%", str(c.get("description") or ""))
                return sum(0.5 if x == "<1" else float(x) for x in found)

            return min(question.get("choices") or [], key=lambda c: (support(c), c["value"]))["value"]
        if behaviour == "first":
            return values[0]
        return marked[0] if marked else values[0]         # mix

    def _answers(self, behaviour: str, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for q in body.get("questions") or []:
            kind, name = q.get("type"), q.get("name")
            if behaviour == "refusal":
                out.append({"type": "refusal", "name": name})
            elif kind == "predicate":
                out.append({"type": "predicate", "name": name, "probability": 0.99})
            elif kind == "choice":
                pick = self._pick(behaviour, q)
                values = [c["value"] for c in q["choices"]]
                rest = [v for v in values if v != pick]
                probabilities = [{"value": v, "probability": 0.9 if v == pick else round(0.1 / max(1, len(rest)), 4)} for v in values]
                answer = {"type": "choice", "name": name, "choice": pick, "probabilities": probabilities, "confidence": 0.85}
                if behaviour == "no_choice":
                    answer.pop("choice")
                if behaviour == "wrong_name":
                    answer["name"] = str(name) + "_other"
                out.append(answer)
            elif kind == "score":
                levels = q.get("levels") or []
                out.append({"type": "score", "name": name, "score": 1.0, "confidence": 0.5,
                            "probabilities": [{"value": i, "label": lv["label"], "probability": round(1.0 / len(levels), 4)}
                                              for i, lv in enumerate(levels)]})
        return out

    def _handle(self, h: Any) -> None:
        began = time.time()
        try:
            length = int(h.headers.get("Content-Length") or 0)
            raw = h.rfile.read(length) if length else b""
        except Exception:
            raw = b""
        auth = h.headers.get("Authorization")
        record: Dict[str, Any] = {"t": round(began, 3), "method": "POST", "path": h.path,
                                  "content_type": h.headers.get("Content-Type"),
                                  "authorization": "missing" if auth is None else ("ok" if auth == "Bearer " + self.key else "wrong"),
                                  "headers": sorted(k.lower() for k in h.headers.keys()), "problems": []}
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = None
            record["problems"].append("the body is not JSON")
        record["body"] = body
        status, payload, raw_out, ctype, behaviour = 200, None, None, "application/json", ""
        if h.path.split("?", 1)[0] == RESPONSES_PATH:
            self._handle_responses(h, record, body, raw, auth, began)
            return
        if h.path.split("?", 1)[0] != PATH:
            status, payload = 404, self._error(404, "Invalid URL (POST %s)" % h.path)
            record["problems"].append("path %s" % h.path)
        elif record["authorization"] != "ok":
            given = (auth or "")[len("Bearer "):] if (auth or "").startswith("Bearer ") else (auth or "")
            status = 401
            payload = self._error(401, None if auth else "You didn't provide an API key.", fill=given)
        elif not str(record["content_type"] or "").lower().startswith("application/json"):
            status, payload = 400, self._error(400, "Content-Type must be application/json")
            record["problems"].append("content type %s" % record["content_type"])
        else:
            record["problems"] += check_request(body)
            if record["problems"]:
                status, payload = 400, self._error(400, "Invalid request: " + "; ".join(record["problems"][:3]))
            elif body.get("model") != MODEL:
                status, payload = 404, self._error(404, "The model `%s` does not exist or you do not have access to it." % body.get("model"))
                record["problems"].append("model %r" % body.get("model"))
        if status == 200:
            names = [q.get("name") for q in body["questions"]]
            check = "sum" in names or "two_plus_two" in names or "sample" in names      # ladder.py --luna-check's questions
            behaviour = self._behaviour(check)
            if not (check and not self.fail_checks):
                low, high = self.delay
                wait = self._rng.uniform(low, high) if high > 0 else 0.0
                if behaviour.startswith("delay:"):
                    wait += float(behaviour.split(":", 1)[1])
                    behaviour = self.mode
                if behaviour == "hang":
                    wait += self.hang_s
                    behaviour = self.mode
                    record["hung"] = True
                if wait > 0:
                    time.sleep(wait)
            if behaviour.startswith("http_"):
                status = int(behaviour.split("_", 1)[1])
                payload = self._error(status, fill=self.key if status == 401 else "")
            elif behaviour == "html":
                status, ctype = 502, "text/html"
                raw_out = b"<!DOCTYPE html><html><head><title>502 Bad Gateway</title></head><body><h1>502 Bad Gateway</h1></body></html>"
            elif behaviour == "malformed_json":
                raw_out = b'{"answers": [{"type": "choice", "name": "action", "cho'
            elif behaviour == "malformed_shape":
                payload = {"answer": {"type": "choice", "choice": "a0"}, "note": "no answers list"}
            elif behaviour == "reset":
                record.update(behaviour=behaviour, status=None)
                self._keep(record)
                try:
                    h.close_connection = True
                    h.connection.shutdown(2)
                    h.connection.close()
                except Exception:
                    pass
                return
            else:
                payload = {"answers": self._answers(behaviour, body)}
                if self.usage:
                    payload["usage"] = {"input_tokens": max(1, len(raw) // 4)}
        record.update(behaviour=behaviour or "rejected", status=status, ms=round((time.time() - began) * 1000.0, 1))
        self._keep(record)
        h._send(status, payload, raw=raw_out, content_type=ctype)

    def _handle_responses(self, h: Any, record: Dict[str, Any], body: Any, raw: bytes, auth: Optional[str], began: float) -> None:
        """POST /v1/responses: the same behaviours, the Responses API's shapes."""
        status, payload, raw_out, ctype, behaviour = 200, None, None, "application/json", ""
        record["api"] = "responses"
        if record["authorization"] != "ok":
            given = (auth or "")[len("Bearer "):] if (auth or "").startswith("Bearer ") else (auth or "")
            status = 401
            payload = self._error(401, None if auth else "You didn't provide an API key.", fill=given)
        elif not str(record["content_type"] or "").lower().startswith("application/json"):
            status, payload = 400, self._error(400, "Content-Type must be application/json")
            record["problems"].append("content type %s" % record["content_type"])
        else:
            record["problems"] += check_responses_request(body)
            if record["problems"]:
                status, payload = 400, self._error(400, "Invalid request: " + "; ".join(record["problems"][:3]))
            elif body.get("model") != SOL_MODEL:
                status, payload = 404, self._error(404, "The model `%s` does not exist or you do not have access to it." % body.get("model"))
                record["problems"].append("model %r" % body.get("model"))
        effort = ((body or {}).get("reasoning") or {}).get("effort") if isinstance(body, dict) else None
        record["effort"] = effort
        if status == 200 and effort in self.refuse_efforts:
            status = 400
            payload = {"error": {"message": "Unsupported value: '%s' is not supported with the '%s' model. Supported values are: 'low', "
                                            "'medium', 'high', 'xhigh', and 'max'." % (effort, SOL_MODEL),
                                 "type": "invalid_request_error", "param": "reasoning.effort", "code": "unsupported_value"}}
            behaviour = "effort_refused"
        elif status == 200:
            questions = questions_of_responses(body)
            record["questions"] = questions
            names = [q.get("name") for q in questions]
            check = "sum" in names or "two_plus_two" in names or "sample" in names
            behaviour = self._behaviour(check)
            if not (check and not self.fail_checks):
                low, high = self.delay
                wait = self._rng.uniform(low, high) if high > 0 else 0.0
                if behaviour.startswith("delay:"):
                    wait += float(behaviour.split(":", 1)[1])
                    behaviour = self.mode
                if behaviour == "hang":
                    wait += self.hang_s
                    behaviour = self.mode
                    record["hung"] = True
                if effort not in (None, "none"):
                    wait += self.reasoning_s
                if wait > 0:
                    time.sleep(wait)
            if behaviour.startswith("http_"):
                status = int(behaviour.split("_", 1)[1])
                payload = self._error(status, fill=self.key if status == 401 else "")
            elif behaviour == "html":
                status, ctype = 502, "text/html"
                raw_out = b"<!DOCTYPE html><html><head><title>502 Bad Gateway</title></head><body><h1>502 Bad Gateway</h1></body></html>"
            elif behaviour == "malformed_json":
                raw_out = b'{"id": "resp_fake", "output": [{"type": "message", "cont'
            elif behaviour == "malformed_shape":
                payload = {"id": "resp_fake", "status": "completed", "output": "no list here"}
            elif behaviour == "reset":
                record.update(behaviour=behaviour, status=None)
                self._keep(record)
                try:
                    h.close_connection = True
                    h.connection.shutdown(2)
                    h.connection.close()
                except Exception:
                    pass
                return
            else:
                reasoning = 0 if effort in (None, "none") else 96
                usage = {"input_tokens": max(1, len(raw) // 4), "input_tokens_details": {"cached_tokens": 0},
                         "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": reasoning}}
                output: List[Dict[str, Any]] = []
                if reasoning:
                    output.append({"type": "reasoning", "id": "rs_fake", "summary": []})
                payload = {"id": "resp_fake_%06d" % len(self.requests), "object": "response", "model": SOL_MODEL, "status": "completed",
                           "output": output, "usage": usage}
                if behaviour == "refusal":
                    content = [{"type": "refusal", "refusal": "I'm sorry, I cannot assist with that request."}]
                elif behaviour == "incomplete":
                    payload.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
                    content = None
                else:
                    obj: Dict[str, Any] = {}
                    for answer in self._answers(behaviour, {"questions": questions}):
                        name = answer.get("name")
                        if answer.get("type") == "choice":
                            item: Dict[str, Any] = {"reason": "The fake's reason for %s." % answer.get("choice"), "confidence": 0.85}
                            if "choice" in answer:
                                item["choice"] = answer["choice"]
                            obj[name] = item
                        else:
                            obj[name] = {"probability": answer.get("probability")}
                    text = json.dumps(obj)
                    content = [{"type": "output_text", "text": text, "annotations": []}]
                    usage["output_tokens"] = reasoning + max(1, len(text) // 4)
                if content is not None:
                    output.append({"type": "message", "id": "msg_fake", "role": "assistant", "status": "completed", "content": content})
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        record.update(behaviour=behaviour or "rejected", status=status, ms=round((time.time() - began) * 1000.0, 1))
        self._keep(record)
        h._send(status, payload, raw=raw_out, content_type=ctype)

    def _keep(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self.requests.append(record)
            key = str(record.get("behaviour"))
            self.counts[key] = self.counts.get(key, 0) + 1
            if self.log_path:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                except OSError:
                    pass

    def game_requests(self) -> List[Dict[str, Any]]:
        """The recorded requests that asked a game's question (not the check's)."""
        out = []
        for r in self.requests:
            asked = r.get("questions") if r.get("api") == "responses" else (
                (r.get("body") or {}).get("questions") if isinstance(r.get("body"), dict) else None)
            names = [q.get("name") for q in asked or []]
            if "action" in names:
                out.append(r)
        return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A fake of OpenAI's Decisions API for tests (the key it expects: FAKE_LUNA_KEY).")
    ap.add_argument("--port", type=int, default=11450)
    ap.add_argument("--mode", default="mix")
    ap.add_argument("--delay", type=float, nargs=2, default=(0.0, 0.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--error-rate", type=float, default=0.0)
    ap.add_argument("--errors", default=",".join(DEFAULT_ERRORS))
    ap.add_argument("--hang-s", type=float, default=5.0)
    ap.add_argument("--usage", action="store_true", help="answers carry a usage object")
    ap.add_argument("--fail-checks", action="store_true", help="the check question is treated like any other")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--switch-after", type=int, default=None, help="after this many game questions, answer every one with --switch-mode")
    ap.add_argument("--switch-mode", default="http_401")
    ap.add_argument("--log", default=None, help="one JSON line per request (never the key)")
    args = ap.parse_args(argv)
    key = os.environ.get("FAKE_LUNA_KEY") or ""
    if not key:
        print("set FAKE_LUNA_KEY to the key the fake expects", flush=True)
        return 2
    fake = FakeDecisions(key=key, port=args.port, mode=args.mode, delay=tuple(args.delay), error_rate=args.error_rate,
                         errors=[e for e in args.errors.split(",") if e], seed=args.seed, usage=args.usage, hang_s=args.hang_s,
                         log_path=args.log, fail_checks=args.fail_checks, switch_after=args.switch_after,
                         switch_mode=args.switch_mode).start()
    print("fake Decisions API at %s (mode %s, delay %s, error rate %.2f)" % (fake.url, args.mode, tuple(args.delay), args.error_rate), flush=True)
    import signal

    def stop(*_a: Any) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, stop)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        while True:
            time.sleep(5.0)
    except KeyboardInterrupt:
        pass
    finally:
        fake.stop()
        print("requests: %d %s" % (len(fake.requests), json.dumps(fake.counts, sort_keys=True)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
