"""
A model decides: OpenAI's Luna (the Decisions API, gpt-6-luna) or Sol (the Responses API,
gpt-6-sol) picks our action, every turn.

    python ladder.py --player luna                 Luna in every game
    python ladder.py --player sol                  Sol in every game
    python ladder.py --arms mix,luna               Luna and today's bot (mix), game by game, balanced
    python ladder.py --arms luna,sol               the two models against each other, game by game
    python ladder.py --player sol-lines            Sol, with the search's favourite actions read 5 turns ahead
    python ladder.py --luna-check                  one tiny real question: the key, the endpoint, the answer's shape
    python ladder.py --luna-check --model sol      the same for Sol

Sol (MODELS["sol"]) is a general model, larger and slower than Luna: it is asked through

    POST https://api.openai.com/v1/responses
    {"model": "gpt-6-sol", "reasoning": {"effort": "none"}, "store": false, "max_output_tokens": 400,
     "input": [{"role": "system", "content": "<how to answer, and the instructions>"},
               {"role": "user", "content": "<the brief, then the choices>"}],
     "text": {"format": {"type": "json_schema", "name": "answers", "strict": true, "schema": {...}}}}
    -> {"output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"action\": {\"reason\": \"...\",
        \"choice\": \"a0-earthquake\", \"confidence\": 0.7}}"}]}], "usage": {"input_tokens": ..., "output_tokens": ...}}

with the same brief and the same choices as Luna's question (responses_body), and its answer is
read into the Decisions shape (responses_answers), so everything below the client is one code
path. Its reasoning effort is "none" (--sol-effort): no hidden reasoning, a short "reason" in the
answer itself. When the API refuses that effort (HTTP 400 naming it) "low" is used for the rest
of the run, said once. Sol costs $2 per 1M input tokens and $10 per 1M output tokens (about half a
cent a decision); the cost is counted from the API's own "usage".

The rest of this text is about Luna; it holds for Sol word for word except where a number is Luna's.

The arm "luna" of search_player.py is today's mix with one more step. At every decision with more
than one legal action the player computes what mix computes (the network's probabilities and win
estimate, Foul Play's search shares and results per action, the read of the opponent), writes it
down as a brief in plain words, and asks Luna ONE choice question whose choices are exactly our
legal actions. Luna's choice is played.

The API (https://developers.openai.com/api/docs/guides/decisions, public beta since 2026-10-06):

    POST https://api.openai.com/v1/decisions
    Authorization: Bearer KEY            Content-Type: application/json
    {"model": "gpt-6-luna", "input": "<the brief>",
     "questions": [{"type": "choice", "name": "action", "instructions": "...",
                    "choices": [{"value": "a0-earthquake", "description": "..."}, ...]}]}
    -> {"answers": [{"type": "choice", "name": "action", "choice": "a0-earthquake",
                     "probabilities": [{"value": "a0-earthquake", "probability": 0.9}, ...], "confidence": 0.8}]}
       (or an answer of type "refusal")

The documented answer has no "usage": the input tokens of a request are estimated here
(estimate_tokens), and read from "usage" when the API does send one. Input costs $0.10 per 1M
tokens and nothing else is charged. Plain HTTPS with requests: no SDK.

It never costs a game (the rails; each one is counted, and written in the decision's note):

    no time      under LunaSettings.min_wait s are left before the decision's deadline: not asked
    timeout      no answer within min(--luna-timeout 1.2 s, the time left before the deadline)
    http_NNN     the API answered with an error (401, 429, 500...), after one retry of a 5xx
    network      the connection failed (one retry when there is time)
    refusal      an answer of type "refusal"
    malformed    an answer that is not the documented shape
    illegal      a choice that is not one of the values we offered
    paused       5 API failures in a row: not asked for 60 s
    key_refused  the API refused the key (401, 403) or does not know the model (404): not asked
                 again in this run, and ladder.py starts no more games
  -> in every one of these, mix's pick is played.
    the blunder guard (on unless --luna-guard off): Luna's pick has under 1% of the network's
    probability AND under 1% of the search's share while mix's pick has over 60% of either:
    mix's pick is played and the decision is marked "overruled".

The key. Read from the environment variable OPENAI_API_KEY, else typed once at a hidden prompt
(getpass). It lives in this process's memory and goes into the Authorization header of the
requests, nowhere else: never a file, a log, a note, a command line or the console. Whatever is
printed about a failure is the HTTP status and the API's own message, passed through redact()
(the key itself, anything shaped like a key, a Bearer token, an Authorization header: ***).
The endpoint is OpenAI's unless it is a test server on this very machine (127.0.0.1): the key is
never sent to any other host.

The brief holds only what our side of the battle legitimately sees: our own team, what the
opponent has shown, and guesses marked as guesses (the random battle sets are public; Foul Play
samples the opponent's hidden sets from them). No user names and no chat: nothing a person typed
reaches the model.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import getpass
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "LUNA_ARM", "DEFAULT_URL", "DEFAULT_MODEL", "KEY_ENV", "TOKEN_BUDGET", "USD_PER_MILLION_TOKENS", "QUESTION_NAME",
    "LunaError", "LunaReply", "LunaClient", "LunaSettings", "Luna", "Facts", "Built", "read_key", "remember_secret",
    "redact", "estimate_tokens", "request_tokens", "parse_answer", "blunder_guard", "build_request", "render_events",
    "check_body", "luna_check", "new_counts", "add_counts", "result_fields", "counts_text", "endpoint_problem",
    "TurnLog", "EVENT_KINDS", "sample_body", "SAMPLE_FILE", "SAMPLE_NAME",
    "ModelSpec", "MODELS", "RESPONSES_URL", "responses_body", "responses_answers", "LINES_NOTE", "cost_usd",
    "SOLO_INSTRUCTIONS", "solo_rule",
]

LUNA_ARM = "luna"
DEFAULT_URL = "https://api.openai.com/v1/decisions"
DEFAULT_MODEL = "gpt-6-luna"
KEY_ENV = "OPENAI_API_KEY"
QUESTION_NAME = "action"
TOKEN_BUDGET = 2500                 # the brief, the instructions and the choices together stay under about this
USD_PER_MILLION_TOKENS = 0.10       # gpt-6-luna: input tokens only
ACTIONS = 14
HERE = os.path.dirname(os.path.abspath(__file__))
RESPONSES_URL = "https://api.openai.com/v1/responses"
LINES_TOKENS = 700                  # on top of TOKEN_BUDGET when the brief carries look-ahead lines


@dataclass(frozen=True)
class ModelSpec:
    """
    A model that can decide. name: its arm family and the player's `deciders` key; api:
    "decisions" (POST /v1/decisions) or "responses" (POST /v1/responses with a JSON schema);
    usd_*: dollars per 1M input, cached input and output tokens; timeout: the seconds a game gives
    one answer; effort: the Responses API's reasoning.effort ("" for the Decisions API).
    """
    name: str
    label: str
    model: str
    api: str
    url: str
    usd_in: float
    usd_cached: float
    usd_out: float
    timeout: float
    effort: str = ""


MODELS: Dict[str, ModelSpec] = {
    "luna": ModelSpec("luna", "Luna", DEFAULT_MODEL, "decisions", DEFAULT_URL, USD_PER_MILLION_TOKENS, USD_PER_MILLION_TOKENS, 0.0, 1.2),
    # https://developers.openai.com/api/docs/models/gpt-6-sol: $2 / $0.20 cached / $10 per 1M tokens;
    # reasoning.effort none, low, medium (the API's default), high, xhigh, max.
    "sol": ModelSpec("sol", "Sol", "gpt-6-sol", "responses", RESPONSES_URL, 2.0, 0.20, 10.0, 5.0, effort="none"),
}


def cost_usd(spec: Optional[ModelSpec], input_tokens: Any, output_tokens: Any = 0, cached_tokens: Any = 0) -> float:
    """What a request cost: the uncached input, the cached input and the output at the model's prices (Luna's without a spec)."""
    spec = spec or MODELS["luna"]
    tokens_in, tokens_out = max(0, int(input_tokens or 0)), max(0, int(output_tokens or 0))
    cached = max(0, min(tokens_in, int(cached_tokens or 0)))
    return ((tokens_in - cached) * spec.usd_in + cached * spec.usd_cached + tokens_out * spec.usd_out) / 1e6


# =============================================================================================
# The key, and keeping it out of everything that is printed or written
# =============================================================================================

_SECRETS: List[str] = []
_KEY_LIKE = re.compile(r"\bsk-[A-Za-z0-9_\-\*]{6,}")
_BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9_\-\.=\*:+/~]{4,}")
_AUTH_HEADER = re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)[^'\",}\r\n]+")


def remember_secret(secret: Optional[str]) -> None:
    """A text that must never be printed or written (the API key): redact() puts *** in its place."""
    if secret and len(secret) >= 6 and secret not in _SECRETS:
        _SECRETS.append(secret)


def redact(text: Any) -> str:
    """
    The text with the key (every remembered secret), anything shaped like an OpenAI key (also the
    half-masked echo in the API's own "Incorrect API key provided: sk-ab***yz"), a Bearer token
    and an Authorization header put as ***.
    """
    out = str(text)
    for secret in _SECRETS:
        if secret in out:
            out = out.replace(secret, "***")
    out = _AUTH_HEADER.sub(r"\1***", out)
    out = _BEARER.sub(r"\1 ***", out)
    out = _KEY_LIKE.sub("sk-***", out)
    return out


def _one_line(text: Any, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


class LunaError(Exception):
    """
    A request to the Decisions API that gave no usable answer. kind: "timeout", "http" (status
    holds the HTTP status), "network", "malformed", "key" (no key, or one that cannot be a key),
    "endpoint" (an address the key must not be sent to). The text is redacted when it is made.
    """

    def __init__(self, kind: str, text: str, status: Optional[int] = None, attempts: int = 1, latency_ms: Optional[float] = None):
        self.kind = str(kind)
        self.text = redact(_one_line(text, 300))
        self.status = status
        self.attempts = int(attempts)
        self.latency_ms = latency_ms
        super().__init__(self.text)

    @property
    def reason(self) -> str:
        """The rail's name in the counts: "timeout", "http_429", "network", "malformed"."""
        return "http_%d" % self.status if self.kind == "http" and self.status else self.kind


def read_key(environ: Optional[Dict[str, str]] = None, ask: Optional[Callable[[str], str]] = None) -> str:
    """
    The API key: OPENAI_API_KEY when it is set, else typed once at a hidden prompt. Remembered
    for redact(). LunaError("key") when there is none or it cannot be one (a paste with spaces or
    a line break in it). Never printed.
    """
    environ = os.environ if environ is None else environ
    key = str(environ.get(KEY_ENV) or "").strip().strip("\"'")
    if not key:
        try:
            key = str((ask or getpass.getpass)("OpenAI API key for Luna (not shown, not stored): ") or "").strip().strip("\"'")
        except EOFError:           # no terminal to type at (a scheduled start, a closed input): no key, said in one line
            key = ""
    if not key:
        raise LunaError("key", "no OpenAI API key: set %s or type the key at the prompt" % KEY_ENV)
    if any(ord(c) < 33 or ord(c) > 126 for c in key):
        remember_secret(key)
        raise LunaError("key", "the key has a space, a line break or another character a key cannot have: paste it again")
    remember_secret(key)
    return key


def endpoint_problem(url: str) -> str:
    """
    Why the key must not be sent to this address, or "". OpenAI's endpoint (https, api.openai.com)
    and a test server on this machine (127.0.0.1, localhost) are the only ones.
    """
    from urllib.parse import urlparse

    try:
        parts = urlparse(str(url))
    except Exception:
        return "not an address"
    host = (parts.hostname or "").lower()
    if parts.scheme == "https" and host == "api.openai.com":
        return ""
    if parts.scheme in ("http", "https") and host in ("127.0.0.1", "localhost", "::1"):
        return ""
    return "the key is sent to OpenAI (https://api.openai.com) or to a test server on this machine only, not to %s" % (host or "?")


# =============================================================================================
# The client
# =============================================================================================

@dataclass
class LunaReply:
    """One answer of the API: the parsed JSON's "answers", how long it took, and the tokens when the API said."""
    answers: Any
    latency_ms: float
    attempts: int = 1
    input_tokens: Optional[int] = None
    status: int = 200
    request_id: Optional[str] = None
    body: Any = None
    output_tokens: Optional[int] = None       # the Responses API's usage (Sol); the Decisions API charges input only
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost_usd: Optional[float] = None          # set by a client that prices the request itself (cache writes, another model's rates)
    served_by: Optional[str] = None           # the API's model id when another model than the one asked gave the answer


RESPONSES_SYSTEM = (
    "Answer with one JSON object that matches the schema, and nothing else. For a choice: first \"reason\", the deciding "
    "fact in at most 25 words; then \"choice\", exactly one of the offered values (the text before the colon); then "
    "\"confidence\", from 0 to 1, that it is the best choice. For a yes/no question: \"probability\", from 0 to 1, that "
    "the answer is yes."
)


def responses_body(body: Dict[str, Any], effort: str = "none") -> Dict[str, Any]:
    """
    A Decisions request (the model, the input, the questions) as a Responses API request that asks
    the same: the input and each question's choices as the user's message, the instructions as
    the system's (a single question's own; several questions carry theirs in the user's message),
    and a strict JSON schema with one object per question, named as the question: a choice is
    {"reason", "choice" (an enum of the offered values), "confidence"}, a predicate {"probability"}.
    """
    questions = [q for q in body.get("questions") or [] if isinstance(q, dict)]
    system = RESPONSES_SYSTEM
    parts: List[str] = []
    source = body.get("input")
    parts.append(source if isinstance(source, str) else json.dumps(source, ensure_ascii=False))
    properties: Dict[str, Any] = {}
    required: List[str] = []
    single = len(questions) == 1
    for q in questions:
        name = str(q.get("name") or "answer")
        instructions = str(q.get("instructions") or "")
        if single and instructions:
            system = instructions + "\n\n" + RESPONSES_SYSTEM
        if q.get("type") == "choice":
            choices = [c for c in q.get("choices") or [] if isinstance(c, dict)]
            head = "QUESTION \"%s\"%s\nCHOICES:" % (name, "" if single or not instructions else ": " + instructions)
            parts.append(head + "\n" + "\n".join("- %s: %s" % (c.get("value"), c.get("description")) for c in choices))
            properties[name] = {"type": "object", "additionalProperties": False, "required": ["reason", "choice", "confidence"],
                                "properties": {"reason": {"type": "string"},
                                               "choice": {"type": "string", "enum": [str(c.get("value")) for c in choices]},
                                               "confidence": {"type": "number"}}}
        else:
            parts.append("QUESTION \"%s\" (yes or no)%s" % (name, "" if single or not instructions else ": " + instructions))
            properties[name] = {"type": "object", "additionalProperties": False, "required": ["probability"],
                                "properties": {"probability": {"type": "number"}}}
        required.append(name)
    out: Dict[str, Any] = {
        "model": body.get("model"),
        "input": [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}],
        "text": {"format": {"type": "json_schema", "name": "answers", "strict": True,
                            "schema": {"type": "object", "additionalProperties": False, "required": required, "properties": properties}}},
        # Without reasoning the answer is some 60 tokens; with it the reasoning's tokens count too.
        "max_output_tokens": 400 if str(effort or "none") == "none" else 6000,
        "store": False,
    }
    if effort:
        out["reasoning"] = {"effort": str(effort)}
    return out


def responses_answers(data: Any, body: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    (the "answers" of a Responses API reply in the Decisions shape, "") or (None, why it has none).
    The reply's text is the JSON object of responses_body's schema; a refusal becomes an answer of
    type "refusal" for every question.
    """
    if not isinstance(data, dict):
        return None, "the reply is not a JSON object"
    texts: List[str] = []
    refused = False
    for item in data.get("output") if isinstance(data.get("output"), list) else []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") if isinstance(item.get("content"), list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif part.get("type") == "refusal":
                refused = True
    if not texts and isinstance(data.get("output_text"), str) and data["output_text"]:
        texts.append(data["output_text"])
    questions = [q for q in body.get("questions") or [] if isinstance(q, dict)]
    if refused and not texts:
        return [{"type": "refusal", "name": q.get("name")} for q in questions], ""
    if not texts:
        why = ""
        if isinstance(data.get("incomplete_details"), dict):
            why = ", " + _one_line(data["incomplete_details"].get("reason"), 40)
        return None, "no output text (status %s%s)" % (_one_line(data.get("status"), 20), why)
    try:
        obj = json.loads("".join(texts))
    except ValueError:
        return None, "the output text is not JSON (%d characters, status %s)" % (len("".join(texts)), _one_line(data.get("status"), 20))
    if not isinstance(obj, dict):
        return None, "the output is not a JSON object"
    answers: List[Dict[str, Any]] = []
    for q in questions:
        name = q.get("name")
        got = obj.get(name)
        if not isinstance(got, dict):
            return None, "no answer named %s in the output" % _one_line(name, 30)
        if q.get("type") == "choice":
            answers.append({"type": "choice", "name": name, "choice": got.get("choice"), "confidence": got.get("confidence"),
                            "probabilities": [], "reason": got.get("reason")})
        else:
            answers.append({"type": "predicate", "name": name, "probability": got.get("probability")})
    return answers, ""


class LunaClient:
    """
    The Decisions API over plain HTTPS (requests), with one connection pool kept open between
    decisions (a new TLS connection would cost more than the answer).

    ask(body, timeout)       from the event loop: the request runs in a thread of this client's
                             own pool and is given up after `timeout` seconds, whatever the socket does.
    ask_sync(body, timeout)  the request itself. One retry (retries=1) of a connection that failed
                             or an HTTP 5xx, when at least retry_min_left s of the timeout are left.
                             Never of a timeout, a 4xx or a 429.
    available()              (False, "paused") for breaker_pause s after breaker_failures failures
                             in a row; (False, "key_refused") for the rest of the run once the API
                             refused the key (401, 403) or the model (404).
    succeeded() / failed(e)  the caller says how each request went (the event loop's side only).

    The key is held in memory and put into the Authorization header of each request; repr() and
    every error text are without it.
    """

    RETRY_STATUS = (500, 502, 503, 504)
    FATAL_STATUS = (401, 403, 404)

    def __init__(self, key: str, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, retries: int = 1,
                 retry_min_left: float = 0.30, breaker_failures: int = 5, breaker_pause: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, say: Optional[Callable[[str], None]] = None,
                 workers: int = 8, api: str = "decisions", effort: str = "", label: str = "Luna",
                 spec: Optional[ModelSpec] = None):
        why = endpoint_problem(url)
        if why:
            raise LunaError("endpoint", why)
        if not key:
            raise LunaError("key", "no OpenAI API key")
        remember_secret(key)
        self._key = key
        self.url = str(url)
        self.model = str(model)
        # Which API the address speaks: "decisions" (Luna) or "responses" (Sol: the same question
        # as a Responses request, responses_body, its answer read back by responses_answers).
        self.api = "responses" if str(api) == "responses" else "decisions"
        self.effort = str(effort or "")
        self.effort_fell_back: Optional[str] = None      # the effort the API refused, once it did
        self.label = str(label or "Luna")
        self.spec = spec
        self.retries = max(0, int(retries))
        self.retry_min_left = float(retry_min_left)
        self.breaker_failures = max(1, int(breaker_failures))
        self.breaker_pause = float(breaker_pause)
        self._clock = clock
        self.say = say or (lambda text: print(text, flush=True))
        self._failures = 0
        self._paused_until: Optional[float] = None
        self.pauses = 0
        self.disabled: Optional[str] = None          # why Luna is off for the rest of the run (the key was refused)
        self.off_reason = "key_refused"              # the rail's name while disabled ("spend_cap": ladder.py's --max-model-usd)
        self.last_error: Optional[str] = None
        self.requests_sent = 0
        self._session: Any = None
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="luna")

    def __repr__(self) -> str:
        return "LunaClient(url=%r, model=%r, key=***)" % (self.url, self.model)

    def _wire(self, body: Dict[str, Any]) -> bytes:
        """The request as it goes out: the Decisions body itself, or the same question for the Responses API."""
        out = body if self.api == "decisions" else responses_body(body, self.effort)
        return json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _usage(data: Dict[str, Any]) -> Dict[str, Optional[int]]:
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}

        def number(value: Any) -> Optional[int]:
            return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

        details_in = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
        details_out = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
        return {"input": next((number(usage.get(k)) for k in ("input_tokens", "prompt_tokens", "total_tokens")
                               if number(usage.get(k)) is not None), None),
                "output": next((number(usage.get(k)) for k in ("output_tokens", "completion_tokens") if number(usage.get(k)) is not None), None),
                "cached": number(details_in.get("cached_tokens")), "reasoning": number(details_out.get("reasoning_tokens"))}

    # -- the request --------------------------------------------------------------------------------

    def _http(self) -> Any:
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter

            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=2, pool_maxsize=8, max_retries=0)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._session = session
        return self._session

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer " + self._key, "Content-Type": "application/json"}

    @staticmethod
    def _api_message(status: int, text: str) -> str:
        """The API's own words about an error: its JSON's error.message (with type and code), else the start of the body."""
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            err = data["error"]
            more = ", ".join("%s %s" % (k, err[k]) for k in ("type", "code", "param") if err.get(k))
            return _one_line(err.get("message") or "no message", 220) + (" (%s)" % more if more else "")
        if isinstance(data, dict) and data.get("error"):
            return _one_line(data.get("error"), 220)
        if re.search(r"<\s*(!doctype|html|head|body)\b", (text or "")[:2000], re.I):
            return "a web page where the API's JSON should be"
        return _one_line(text, 160) or "no body"

    def ask_sync(self, body: Dict[str, Any], timeout: float) -> LunaReply:
        """One question, answered within `timeout` seconds or LunaError. Blocking: call it from a thread."""
        import requests

        began = time.monotonic()
        deadline = began + float(timeout)
        payload = self._wire(body)
        attempts = 0
        while True:
            attempts += 1
            left = deadline - time.monotonic()
            if left <= 0.01:
                raise LunaError("timeout", "no answer within %.0f ms" % (1000.0 * timeout), attempts=attempts,
                                latency_ms=(time.monotonic() - began) * 1000.0)
            error: LunaError
            again = False
            try:
                self.requests_sent += 1
                # No redirects: the key and the brief go to the address that was checked (endpoint_problem)
                # and to no other. A 3xx comes back as an HTTP error, and mix's pick is played.
                answer = self._http().post(self.url, data=payload, headers=self._headers(), timeout=(left, left),
                                           allow_redirects=False)
                status = int(answer.status_code)
                text = answer.text
            except requests.Timeout:
                raise LunaError("timeout", "no answer within %.0f ms" % (1000.0 * timeout), attempts=attempts,
                                latency_ms=(time.monotonic() - began) * 1000.0)
            except requests.RequestException as exc:
                error = LunaError("network", "%s: %s" % (type(exc).__name__, _one_line(exc, 200)), attempts=attempts,
                                  latency_ms=(time.monotonic() - began) * 1000.0)
                again = True
            else:
                latency = (time.monotonic() - began) * 1000.0
                if status == 200:
                    try:
                        data = json.loads(text)
                    except ValueError:
                        raise LunaError("malformed", "HTTP 200 with a body that is not JSON (%d characters)" % len(text or ""),
                                        status=200, attempts=attempts, latency_ms=latency)
                    if self.api == "responses":
                        answers, why = responses_answers(data, body)
                        if answers is None:
                            raise LunaError("malformed", "HTTP 200 without a usable answer: %s" % why,
                                            status=200, attempts=attempts, latency_ms=latency)
                    elif not isinstance(data, dict) or not isinstance(data.get("answers"), list):
                        raise LunaError("malformed", "HTTP 200 without an \"answers\" list: %s" % _one_line(text, 120),
                                        status=200, attempts=attempts, latency_ms=latency)
                    else:
                        answers = data["answers"]
                    used = self._usage(data)
                    return LunaReply(answers=answers, latency_ms=round(latency, 1), attempts=attempts, input_tokens=used["input"],
                                     status=200, request_id=answer.headers.get("x-request-id"), body=data,
                                     output_tokens=used["output"], cached_tokens=used["cached"], reasoning_tokens=used["reasoning"])
                message = self._api_message(status, text)
                error = LunaError("http", "HTTP %d: %s" % (status, message), status=status,
                                  attempts=attempts, latency_ms=latency)
                again = status in self.RETRY_STATUS
                if (self.api == "responses" and status == 400 and self.effort not in ("", "low")
                        and re.search(r"reasoning|effort", message, re.I)):
                    # The API does not take this effort for this model (or this account): "low" from
                    # here on, said once, and this question is asked again when there is time.
                    self.effort_fell_back = self.effort
                    self.effort = "low"
                    self.say("%s: the API refused reasoning effort \"%s\" (%s): \"low\" is used from here on (answers take longer)."
                             % (self.label, self.effort_fell_back, redact(message)))
                    payload = self._wire(body)
                    attempts -= 1
                    if deadline - time.monotonic() >= self.retry_min_left:
                        continue
            if not again or attempts > self.retries or deadline - time.monotonic() < self.retry_min_left:
                raise error

    async def ask(self, body: Dict[str, Any], timeout: float) -> LunaReply:
        """ask_sync from the event loop: in this client's own threads, given up after `timeout` s whatever the socket does."""
        loop = asyncio.get_running_loop()
        began = loop.time()
        future = loop.run_in_executor(self._pool, self.ask_sync, body, float(timeout))
        try:
            return await asyncio.wait_for(future, timeout=float(timeout) + 0.02)
        except asyncio.TimeoutError:
            raise LunaError("timeout", "no answer within %.0f ms" % (1000.0 * timeout), latency_ms=(loop.time() - began) * 1000.0)

    # -- the breaker --------------------------------------------------------------------------------

    def available(self) -> Tuple[bool, str]:
        if self.disabled:
            return False, str(self.off_reason or "key_refused")
        if self._paused_until is not None:
            if self._clock() < self._paused_until:
                return False, "paused"
            self._paused_until = None
        return True, ""

    def succeeded(self) -> None:
        self._failures = 0

    def failed(self, error: LunaError) -> None:
        """An API failure (a timeout, an HTTP error, a failed connection, an answer of the wrong shape)."""
        self.last_error = error.text
        if error.kind == "http" and error.status in self.FATAL_STATUS:
            if not self.disabled:
                self.disabled = error.text
                self.say("%s is OFF for the rest of this run: the API refused the key or the model (%s). Today's bot (mix) "
                         "plays the games in progress, and no new games are started." % (self.label, error.text))
            return
        self._failures += 1
        if self._failures >= self.breaker_failures:
            self._failures = 0
            self.pauses += 1
            self._paused_until = self._clock() + self.breaker_pause
            self.say("%s: %d API failures in a row (the last: %s). Not asked for %.0f s: today's bot (mix) plays meanwhile."
                     % (self.label, self.breaker_failures, error.text, self.breaker_pause))

    def close(self) -> None:
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass
        try:
            if self._session is not None:
                self._session.close()
        except Exception:
            pass


# =============================================================================================
# Tokens
# =============================================================================================

_PIECE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
# Calibrated on real briefs against the o200k_base encoding (tests/test_luna.py holds it there when
# tiktoken is installed): the estimate stays a little above the encoder's count.
TOKEN_FACTOR = 0.88


def estimate_tokens(text: Any) -> int:
    """
    About how many tokens a text is: a word of up to 4 letters is one, longer ones (names such as
    Corviknight) one per 4 letters, a number one per 3 digits, every other sign one. No tokenizer
    is needed for it; it errs on the high side.
    """
    n = 0
    for piece in _PIECE.findall(str(text or "")):
        if piece[0].isalpha():
            n += max(1, (len(piece) + 3) // 4)
        elif piece[0].isdigit():
            n += (len(piece) + 2) // 3
        else:
            n += 1
    return int(math.ceil(n * TOKEN_FACTOR))


def request_tokens(body: Dict[str, Any]) -> int:
    """The estimate for a whole request body: the input, and every question's instructions, values and descriptions."""
    parts: List[str] = []
    source = body.get("input")
    if isinstance(source, str):
        parts.append(source)
    else:
        for message in source or []:
            for item in (message.get("content") if isinstance(message, dict) else None) or []:
                if isinstance(item, dict) and item.get("type") == "input_text":
                    parts.append(str(item.get("text") or ""))
    extra = 12
    for q in body.get("questions") or []:
        parts.append(str(q.get("name") or ""))
        parts.append(str(q.get("instructions") or ""))
        extra += 8
        for c in q.get("choices") or []:
            parts.append(str(c.get("value") or ""))
            parts.append(str(c.get("description") or ""))
            extra += 4
        for level in q.get("levels") or []:
            parts.append(str(level.get("label") or ""))
            parts.append(str(level.get("description") or ""))
            extra += 4
    return estimate_tokens("\n".join(parts)) + extra


# =============================================================================================
# The answer, and the blunder guard
# =============================================================================================

def parse_answer(answers: Any, name: str, values: Dict[str, int]) -> Dict[str, Any]:
    """
    The choice answer named `name` out of the API's "answers", against the values we offered
    (value -> our action). {"kind": "choice", "action", "value", "confidence", "probabilities":
    {action: p}} when it is one of ours; else {"kind": "refusal" | "illegal" | "malformed", "why"}.
    """
    if not isinstance(answers, list) or not answers:
        return {"kind": "malformed", "why": "no answers"}
    answer = next((a for a in answers if isinstance(a, dict) and a.get("name") == name), None)
    if answer is None:
        # The only answer, without a name of its own (the guide's SDK examples treat a refusal's
        # name as optional; a null is the same as none): it is the answer to our one question.
        answer = answers[0] if len(answers) == 1 and isinstance(answers[0], dict) and answers[0].get("name") is None else None
    if answer is None:
        return {"kind": "malformed", "why": "no answer named %s" % name}
    kind = answer.get("type")
    if kind == "refusal":
        return {"kind": "refusal", "why": "the model refused"}
    if kind != "choice":
        return {"kind": "malformed", "why": "an answer of type %s where a choice was asked" % _one_line(kind, 30)}
    value = answer.get("choice")
    if not isinstance(value, str) or isinstance(value, bool):
        return {"kind": "malformed", "why": "the choice is not a text"}
    if value not in values:
        return {"kind": "illegal", "why": "a choice we did not offer", "value": _one_line(value, 60)}
    confidence = answer.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None
    probabilities: Dict[int, float] = {}
    for row in answer.get("probabilities") or []:
        if not isinstance(row, dict):
            continue
        v, p = row.get("value"), row.get("probability")
        if isinstance(v, str) and v in values and isinstance(p, (int, float)) and not isinstance(p, bool):
            probabilities[values[v]] = round(float(p), 4)
    out = {"kind": "choice", "action": int(values[value]), "value": value,
           "confidence": None if confidence is None else round(confidence, 4), "probabilities": probabilities}
    reason = answer.get("reason")       # Sol's own few words (responses_answers); the Decisions API gives none
    if isinstance(reason, str) and reason.strip():
        out["reason"] = redact(_one_line(reason, 200))
    return out


def blunder_guard(luna_action: int, mix_action: int, net: Sequence[float], shares: Optional[Sequence[float]],
                  low: float = 0.01, high: float = 0.60) -> str:
    """
    Why Luna's pick is overruled, or "". It is when the pick has under `low` of the network's
    probability AND under `low` of the search's share while mix's pick has over `high` of either.
    net: the network's probability per action, a move and its Tera form counted together
    (search_player.merged_prior: the network does not judge Terastallizing). shares: the search's
    share of exactly each action; None when the search gave no answer (then it has no share to
    speak for or against anything).
    """
    if int(luna_action) == int(mix_action):
        return ""
    p_luna, p_mix = float(net[luna_action]), float(net[mix_action])
    s_luna = float(shares[luna_action]) if shares is not None else 0.0
    s_mix = float(shares[mix_action]) if shares is not None else 0.0
    if p_luna < low and s_luna < low and (p_mix > high or s_mix > high):
        return ("Luna's pick has %.1f%% of the network and %.1f%% of the search; mix's pick has %.0f%% and %.0f%%"
                % (100 * p_luna, 100 * s_luna, 100 * p_mix, 100 * s_mix))
    return ""


# =============================================================================================
# What a decision knows: the facts the player hands over, and the request built from them
# =============================================================================================

@dataclass
class Facts:
    """
    One decision as the search player computed it (the same numbers mix plays from).

    mask          the 14 legal-action flags
    actions       legal action -> (poke-env Move or Pokemon, the order's keywords): ObservationAssembler.map_index_to_order
    probs         the network's probabilities over the legal actions (search_player.legal_probabilities)
    net_action    the network's own pick
    mix_action    what mix plays
    shares        the search's share per action, None when it gave no answer
    scores        its mean result per action (0 to 1, None where it did not look)
    search_reason why there is no search answer ("timeout", "down", "rejoined"...), "" when there is one
    iterations    the search's size
    prediction    the opponent model's read: [(name, probability)], likeliest first, in the search's names
                  ("move earthquake", "switch garchomp", "switch ?"); None when there is none
    p_switch, p_tera   of that read
    opponent_options   the search's own read of the opponent [(name, share, seen)], used when there is no model read
    worlds        the sets Foul Play sampled for the opponent's active Pokemon (Advice.opponent_worlds): guesses
    win_est       the network's win estimate (value_fields), 0 to 1
    last_turn     the protocol lines of the turn before this one (split at "|"), this_turn those of this turn so far
    lines         the look-ahead lines of a lines arm (Advice.lines_by_action): our action -> {"value", "end",
                  "turns", "worlds": [{"chance", "value", "turns", "steps": [...]}]}; empty without lines
    revealed      the ids of the opponent's Pokemon shown so far (the active one first): a Pokemon of a line
                  that is not among them is a sampled guess, and is written as one
    """
    mask: Sequence[float]
    actions: Dict[int, Tuple[Any, Dict[str, Any]]]
    probs: Sequence[float]
    net_action: int
    mix_action: int
    shares: Optional[Sequence[float]] = None
    scores: Optional[Sequence[Optional[float]]] = None
    search_reason: str = ""
    iterations: int = 0
    prediction: Optional[List[Tuple[str, float]]] = None
    p_switch: Optional[float] = None
    p_tera: Optional[float] = None
    opponent_options: Optional[List[Tuple[str, float, Optional[bool]]]] = None
    worlds: Optional[List[Dict[str, Any]]] = None
    win_est: Optional[float] = None
    last_turn: List[List[str]] = field(default_factory=list)
    this_turn: List[List[str]] = field(default_factory=list)
    lines: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    revealed: List[str] = field(default_factory=list)
    solo: bool = False           # the model plays alone (the arms luna-solo, sol-solo): no adviser's number is written anywhere,
                                 # and mix_action is the fallback rule's pick (solo_rule), not mix's
    # A model alone has no memory of its own between decisions, so the brief carries it:
    memory: List[Dict[str, Any]] = field(default_factory=list)      # its last moves in this battle, oldest first: {"turn", "text",
                                                                    # "reason", "switch", "forced", "by" ("model" or "rule")}
    earlier: List[Tuple[int, List[List[str]]]] = field(default_factory=list)   # (turn, its protocol lines) of the turns before the last one

    def merged(self) -> List[float]:
        """The network's probability per action with a move and its Tera form counted together."""
        q = [float(p) for p in self.probs]
        for k in range(4):
            if self.mask[k] > 0.5 and self.mask[10 + k] > 0.5:
                q[k] = q[10 + k] = float(self.probs[k]) + float(self.probs[10 + k])
        return q


@dataclass
class Built:
    """A request ready to send: the body, value -> our action, and what it weighs."""
    body: Dict[str, Any]
    values: Dict[str, int]
    brief: str
    question: Dict[str, Any]
    tokens: int
    detail: int = 0                       # 0: the full brief; higher: parts left out to stay under the budget
    problems: List[str] = field(default_factory=list)     # sections that could not be written (left out)
    ms: float = 0.0


INSTRUCTIONS = (
    "You choose our action in a Pokemon Showdown gen 9 random battle. Pick the action most likely to WIN THE GAME, "
    "not only this turn. Each choice lists its facts and two advisors: net = our neural network's probability for "
    "that move or switch (it does not judge Terastallizing), search = the share of visits our look-ahead search gave "
    "exactly that action, with its average result (0 lost, 1 won). Together (\"mix\") they play at about 2350 Elo, "
    "and the choice marked MIX'S PICK is what they would play: follow them unless the facts give a concrete reason "
    "not to. "
    "Respect type immunities and abilities: never pick a move that does nothing. "
    "Do not stay in on a boosted sweeper without an answer: knock it out, remove its boosts, force it out, or switch to "
    "something that takes its hits. "
    "Take a safe knockout when one is there. "
    "When ahead, convert the lead safely: no needless risk, no low-accuracy move where a sure one does the job. "
    "Do not waste Terastallization, and do not switch without a reason: a switch gives the opponent a free turn."
)

SOLO_INSTRUCTIONS = (
    "You are playing a Pokemon Showdown gen 9 random battle, and you alone choose our action: no adviser is behind you. "
    "Pick the action most likely to WIN THE GAME, not only this turn. Each choice lists its facts: what the move does, the "
    "damage it is estimated to do and whether that knocks out, who moves first, what a switch-in takes on the way in. "
    "Use only the numbers and facts given: do not assume an immunity or a resistance that is not written. "
    "Respect type immunities and abilities: never pick a move that does nothing. "
    "Take a safe knockout when one is there. "
    "SWITCHING IS EXPENSIVE: it uses our whole turn, the foe acts freely against the Pokemon coming in, and entry hazards "
    "hurt it. Switch only when BOTH hold: (1) staying in loses the active Pokemon this turn for nothing, or it cannot "
    "hurt the foe; and (2) the Pokemon coming in takes the foe's likely hit well and threatens it back. Otherwise attack. "
    "YOUR LAST MOVES lists what you did on the turns before: keep to your plan, and never switch back to a Pokemon you "
    "just took out, or take out one you just brought in, unless the foe's active Pokemon changed. "
    "A Pokemon that would be knocked out anyway should usually attack first rather than hand the foe a free turn. "
    "FOE'S POSSIBLE MOVES lists what the foe can carry and what each does to our active: plan for the worst of them, "
    "not only for the moves it has shown. "
    "When the foe is boosting or can boost, do not give it free turns: knock it out, cripple it, force it out, or bring "
    "in the Pokemon that takes its hits, now. "
    "Set up, heal or use a status move only when the foe cannot punish it this turn; otherwise attack with the move that does the most. "
    "When ahead, convert the lead safely: no needless risk, no low-accuracy move where a sure one does the job. "
    "Terastallize when it decides a key exchange (it turns a hit into a knockout, or lets us survive one): do not waste it."
)
SOLO_TOKENS = 700                   # on top of TOKEN_BUDGET for a model alone: its memory and the foe's possible moves


def solo_rule(battle: Any, mask: Sequence[float], actions: Dict[int, Tuple[Any, Dict[str, Any]]]) -> int:
    """
    The action of a model-alone arm whenever the model gives none (a timeout, an API error, no
    key): a rule, not the network and not the search. The plain move (never its Tera form) with
    the largest base power x STAB x type multiplier against the foe's active x accuracy; with no
    damaging move the first legal move; on a forced switch the Pokemon with the most HP left.
    """
    legal = [a for a in sorted(actions) if a < len(mask) and mask[a] > 0.5]
    if not legal:
        raise ValueError("no legal action")
    foe = getattr(battle, "opponent_active_pokemon", None)
    me = getattr(battle, "active_pokemon", None)
    mine = [t for t in (getattr(me, "types", None) or ()) if t is not None]
    best, best_score = None, 0.0
    for a in legal:
        if a > 3:
            continue
        move = actions[a][0]
        try:
            power = float(getattr(move, "base_power", 0) or 0)
            if power <= 0 or str(getattr(getattr(move, "category", None), "name", "") or "").upper() == "STATUS":
                continue
            times = 1.0
            if foe is not None:
                times = float(foe.damage_multiplier(move))
            accuracy = getattr(move, "accuracy", 1.0)
            accuracy = 1.0 if accuracy is True or accuracy is None else float(accuracy)
            if accuracy > 1.0:
                accuracy /= 100.0
            score = power * times * (1.5 if getattr(move, "type", None) in mine else 1.0) * accuracy
        except Exception:
            continue
        if score > best_score:
            best, best_score = a, score
    if best is not None:
        return best
    plain = [a for a in legal if a < 4]
    if plain:
        return plain[0]
    switches = [a for a in legal if 4 <= a <= 9]
    if switches:
        return max(switches, key=lambda a: (float(getattr(actions[a][0], "current_hp_fraction", 0.0) or 0.0), -a))
    return legal[0]


LINES_NOTE = (
    " Some choices carry a LINE: our search played that action forward a few turns, both sides choosing by search, the "
    "likeliest outcome of every turn, and \"line\" is its result at the end (0 lost, 1 won); LOOK-AHEAD LINES tells each "
    "one. A line is one likely future, not a certainty: let a clearly better or worse line count, and ignore small "
    "differences."
)

_STATUS = {"brn": "burned", "par": "paralyzed", "psn": "poisoned", "tox": "badly poisoned", "slp": "asleep", "frz": "frozen"}
_BOOST = {"atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe", "accuracy": "Acc", "evasion": "Eva"}
_EFFECTS = {"substitute": "behind a Substitute", "leechseed": "Leech Seeded", "confusion": "confused", "taunt": "Taunted",
            "encore": "Encored", "yawn": "drowsy (asleep next turn)", "perish3": "Perish count 3", "perish2": "Perish count 2",
            "perish1": "Perish count 1", "perish0": "Perish count 0", "curse": "cursed", "saltcure": "Salt Cured",
            "disable": "a move Disabled", "torment": "Tormented", "partiallytrapped": "trapped by a binding move",
            "trapped": "trapped", "attract": "infatuated", "focusenergy": "Focus Energy", "magnetrise": "Magnet Rise",
            "aquaring": "Aqua Ring", "ingrain": "Ingrain", "destinybond": "Destiny Bond up", "healblock": "Heal Blocked",
            "throatchop": "Throat Chopped", "glaiverush": "open after Glaive Rush", "protosynthesis": "Protosynthesis active",
            "quarkdrive": "Quark Drive active", "flashfire": "Flash Fire boosted", "charge": "charged", "smackdown": "grounded",
            "stockpile1": "Stockpile 1", "stockpile2": "Stockpile 2", "stockpile3": "Stockpile 3", "noretreat": "No Retreat",
            "octolock": "Octolocked", "tarshot": "Tar Shot", "syrupbomb": "Syrup Bombed", "embargo": "Embargoed",
            "slowstart": "Slow Start", "truant": "loafing (Truant)"}
_SIDE = {"stealthrock": "Stealth Rock", "spikes": "Spikes", "toxicspikes": "Toxic Spikes", "stickyweb": "Sticky Web",
         "reflect": "Reflect", "lightscreen": "Light Screen", "auroraveil": "Aurora Veil", "tailwind": "Tailwind",
         "safeguard": "Safeguard", "mist": "Mist"}
_WEATHER = {"raindance": "rain", "primordialsea": "heavy rain", "sunnyday": "sun", "desolateland": "harsh sun",
            "sandstorm": "sandstorm", "snowscape": "snow", "hail": "hail", "deltastream": "strong winds"}
_FIELD = {"electricterrain": "Electric Terrain", "grassyterrain": "Grassy Terrain", "mistyterrain": "Misty Terrain",
          "psychicterrain": "Psychic Terrain", "trickroom": "Trick Room (slower Pokemon move first)", "gravity": "Gravity",
          "magicroom": "Magic Room", "wonderroom": "Wonder Room", "neutralizinggas": "Neutralizing Gas"}
# What a move does when the data's fields do not say it (a few words each).
_MOVE_NOTES = {
    "knockoff": "removes the target's item (1.5x damage while it holds one, counted in the estimate)", "rapidspin": "clears hazards from our side, our Spe +1", "mortalspin": "clears hazards from our side, poisons",
    "defog": "clears hazards from both sides and the foe's screens", "tidyup": "clears hazards and Substitutes, our Atk +1 Spe +1",
    "haze": "resets every stat change on both sides", "clearsmog": "resets the target's stat changes",
    "trick": "swaps items with the target", "switcheroo": "swaps items with the target", "painsplit": "shares HP evenly with the target",
    "wish": "heals 50% at the end of next turn", "rest": "full heal, then we sleep 2 turns", "sleeptalk": "uses another of our moves while asleep",
    "bellydrum": "our Atk to +6 for half our HP", "curse": "our Atk +1 Def +1 Spe -1 (as a Ghost: curses the foe for half our HP)",
    "destinybond": "if the foe knocks us out this turn it faints too", "healingwish": "we faint; our next Pokemon in is fully healed",
    "lunardance": "we faint; our next Pokemon in is fully healed", "memento": "we faint; the foe's Atk and SpA -2",
    "partingshot": "the foe's Atk and SpA -1, then we switch out", "courtchange": "swaps the hazards and screens of both sides",
    "superfang": "halves the target's current HP", "ruination": "halves the target's current HP", "naturesmadness": "halves the target's current HP",
    "seismictoss": "fixed damage equal to our level", "nightshade": "fixed damage equal to our level",
    "counter": "returns double the physical damage we took this turn", "mirrorcoat": "returns double the special damage we took this turn",
    "metalburst": "returns 1.5 times the damage we took this turn", "comeuppance": "returns 1.5 times the damage we took this turn",
    "endeavor": "brings the target down to our HP", "finalgambit": "we faint; damage equal to our HP",
    "fakeout": "flinches; works only on our first turn in", "firstimpression": "works only on our first turn in",
    "suckerpunch": "fails unless the target attacks this turn", "thunderclap": "fails unless the target attacks this turn",
    "focuspunch": "fails if we are hit first", "futuresight": "the damage lands two turns later", "doomdesire": "the damage lands two turns later",
    "terablast": "its type is our Tera type once Terastallized", "revivalblessing": "revives a fainted teammate at half HP",
    "bodypress": "uses our Defense for damage", "foulplay": "uses the target's Attack for damage",
    "psyshock": "special, but hits Defense", "psystrike": "special, but hits Defense", "secretsword": "special, but hits Defense",
    "stoneaxe": "also sets Stealth Rock", "ceaselessedge": "also sets a layer of Spikes", "saltcure": "the target loses HP every turn",
    "shedtail": "a Substitute for our next Pokemon, and we switch out", "batonpass": "we switch out and pass our boosts",
    "teleport": "we switch out, moving last", "chillyreception": "sets snow, then we switch out",
    "strengthsap": "heals us by the target's Attack, its Atk -1", "leechseed": "saps the target every turn", "toxic": "badly poisons",
    "protect": "blocks the foe's move this turn", "detect": "blocks the foe's move this turn", "spikyshield": "blocks, and hurts on contact",
    "kingsshield": "blocks, and lowers Atk on contact", "banefulbunker": "blocks, and poisons on contact", "silktrap": "blocks, and lowers Spe on contact",
    "burningbulwark": "blocks, and burns on contact", "substitute": "a Substitute for 25% of our HP", "encore": "locks the target into its last move",
    "taunt": "the target cannot use status moves", "disable": "disables the target's last move", "trickroom": "slower Pokemon move first for 5 turns",
    "perishsong": "every Pokemon on the field faints in 3 turns unless switched", "roar": "forces the target out (moves last)",
    "whirlwind": "forces the target out (moves last)", "dragontail": "forces the target out (moves last)", "circlethrow": "forces the target out (moves last)",
    "explosion": "we faint", "selfdestruct": "we faint", "mistyexplosion": "we faint", "poltergeist": "fails if the target has no item",
    "lastresort": "fails until our other moves were used", "steelbeam": "costs half our HP", "mindblown": "costs half our HP",
    "filletaway": "our Atk SpA Spe +2 for half our HP", "clangoroussoul": "all our stats +1 for a third of our HP", "shellsmash": "our Atk SpA Spe +2, Def SpD -1",
    "noretreat": "all our stats +1, we cannot switch", "transform": "we become a copy of the target", "spore": "puts the target to sleep",
    "yawn": "the target falls asleep at the end of next turn", "soak": "makes the target a pure Water type", "hex": "double power on a statused target",
    "facade": "double power when we are statused", "acrobatics": "double power without an item", "weatherball": "double power and typed by the weather",
    "risingvoltage": "double power on Electric Terrain", "expandingforce": "stronger on Psychic Terrain", "solarbeam": "charges a turn unless the sun is up",
    "meteorbeam": "charges a turn, our SpA +1", "glaiverush": "afterwards we take double damage until our next move",
}


def _id(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(getattr(text, "name", text) or "").lower())


def _cap(text: Any) -> str:
    name = str(getattr(text, "name", text) or "")
    if name.upper() == "THREE_QUESTION_MARKS":
        return "???"
    return name.replace("_", " ").title() if name.isupper() or name.islower() else name


def _pct(x: Any) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "?"
    if 0.0 < x < 0.005:
        return "<1%"
    return "%d%%" % round(100.0 * x)


_NAMES: Optional[Dict[str, Dict[str, Any]]] = None


def _public_sets() -> Dict[str, Dict[str, Any]]:
    """
    The public random battle sets (pokemon-showdown data/random-battles/gen9, the file
    bench/opp_dataset.py reads): {"species": {id: {"items", "abilities", "moves", "tera"}},
    "item_names": {id: name}, "ability_names": {id: name}}. Empty when the file is not there.
    """
    global _NAMES
    if _NAMES is not None:
        return _NAMES
    out: Dict[str, Dict[str, Any]] = {"species": {}, "item_names": {}, "ability_names": {}}
    paths = [os.path.join(HERE, "heuristic", "cache", "gen9randombattle.json"),
             r"C:\Users\you\ps-ppo\heuristic\cache\gen9randombattle.json",
             "/mnt/c/Users/you/ps-ppo/heuristic/cache/gen9randombattle.json"]
    for path in paths:
        try:
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as fh:
                sets = json.load(fh)
            for species, entry in sets.items():
                row: Dict[str, List[str]] = {"items": [], "abilities": [], "moves": [], "tera": []}
                for role in (entry.get("roles") or {}).values():
                    for key, source in (("items", "items"), ("abilities", "abilities"), ("moves", "moves"), ("tera", "teraTypes")):
                        for name in role.get(source) or []:
                            if key == "items":
                                out["item_names"][_id(name)] = str(name)
                            elif key == "abilities":
                                out["ability_names"][_id(name)] = str(name)
                            ident = str(name) if key == "tera" else _id(name)
                            if ident not in row[key]:
                                row[key].append(ident)
                out["species"][_id(species)] = row
            break
        except Exception:
            continue
    try:
        # Every ability's name as the dex writes it (a forme's own ability, Tera Shell say, is in no random battle set).
        import setup_moves as sm

        sm._data()
        for entry in (sm._DEX or {}).values():
            for name in (entry.get("abilities") or {}).values():
                out["ability_names"].setdefault(_id(name), str(name))
    except Exception:
        pass
    try:
        with open(os.path.join(HERE, "randbats_abilities_gen9.json"), encoding="utf-8") as fh:
            for species, abilities in (json.load(fh).get("abilities") or {}).items():
                out["species"].setdefault(_id(species), {"items": [], "abilities": [], "moves": [], "tera": []})
                if not out["species"][_id(species)]["abilities"]:
                    out["species"][_id(species)]["abilities"] = [_id(a) for a in abilities]
    except Exception:
        pass
    _NAMES = out
    return out


def _set_of_species(species: Any) -> Dict[str, List[str]]:
    sets = _public_sets()["species"]
    ident = _id(species)
    if ident in sets:
        return sets[ident]
    try:
        import setup_moves as sm

        sm._data()
        base = _id(((sm._DEX or {}).get(ident) or {}).get("baseSpecies") or "")
        if base and base in sets:
            return sets[base]
    except Exception:
        pass
    return {"items": [], "abilities": [], "moves": [], "tera": []}


def _item_name(item: Any) -> str:
    ident = _id(item)
    return _public_sets()["item_names"].get(ident) or ident


def _ability_name(ability: Any) -> str:
    ident = _id(ability)
    return _public_sets()["ability_names"].get(ident) or ident


class _Board:
    """Everything the brief reads off one decision, worked out once. Every part is optional: what cannot be read is left out."""

    def __init__(self, battle: Any, facts: Facts):
        import setup_counter as sc
        import setup_moves as sm

        self.sc, self.sm = sc, sm
        self.battle, self.facts = battle, facts
        self.problems: List[str] = []
        self.role = getattr(battle, "player_role", None) or "p1"
        self.ours = getattr(battle, "active_pokemon", None)
        self.theirs = getattr(battle, "opponent_active_pokemon", None)
        self.forced = bool(getattr(battle, "force_switch", False))
        self.turn = int(getattr(battle, "turn", 0) or 0)
        self.weather = self._try(lambda: sc._weather(battle), "")
        self.trick_room = any(_id(f) == "trickroom" for f in (getattr(battle, "fields", None) or {}))
        self.our_side = {_id(c): v for c, v in (getattr(battle, "side_conditions", None) or {}).items()}
        self.foe_side = {_id(c): v for c, v in (getattr(battle, "opponent_side_conditions", None) or {}).items()}
        self.worlds = [w for w in (facts.worlds or []) if isinstance(w, dict) and isinstance(w.get("active"), dict)]
        # The opponent's active Pokemon's ability and item as the damage estimates take them: what the
        # battle has shown, else what every set Foul Play sampled agrees on. The brief says which it is
        # (foe_*_shown: the battle's own word, or the only ability the species can have).
        self.foe_ability = self._try(lambda: sc.foe_ability(self.theirs, self.worlds), "") if self.theirs is not None else ""
        self.foe_item = self._try(lambda: sc.foe_item(self.theirs, self.worlds), "") if self.theirs is not None else ""
        self.foe_ability_shown = self._try(lambda: sc.foe_ability(self.theirs, ()), "") if self.theirs is not None else ""
        self.foe_item_shown = self._try(lambda: sc.foe_item(self.theirs, ()), "") if self.theirs is not None else ""
        self.foe_f = None
        if self.theirs is not None:
            self.foe_f = self._try(lambda: sc.fighter_of(self.theirs, ours=False, ability=self.foe_ability,
                                                         item=self.foe_item if self.foe_item != "none" else ""), None)
        self.our_f = self.fighter(self.ours) if self.ours is not None else None
        self.foe_seen = [_id(m) for m in (getattr(self.theirs, "moves", None) or {})] if self.theirs is not None else []
        self.foe_maybe = self._maybe_moves()
        self.foe_hp = float(getattr(self.theirs, "current_hp_fraction", 0.0) or 0.0) if self.theirs is not None else 0.0

    def _try(self, fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception:
            return default

    # -- our Pokemon -------------------------------------------------------------------------------

    def own_tera(self, mon: Any) -> str:
        """Our Pokemon's Tera type from the request ("" when it is not there)."""
        raw = (getattr(mon, "_last_request", None) or {}).get("teraType") or getattr(mon, "tera_type", None)
        return _cap(raw) if raw else ""

    def fighter(self, mon: Any) -> Any:
        f = self.sc.fighter_of(mon, ours=True)
        tera = self.own_tera(mon)
        return replace(f, tera=tera.upper()) if tera else f

    # -- the opponent's hidden things, as guesses ---------------------------------------------------

    def _weighted(self, key: str) -> List[Tuple[str, float]]:
        total, acc = 0.0, {}
        for w in self.worlds:
            chance = float(w.get("chance") or 0.0) or 1.0
            total += chance
            value = _id((w.get("active") or {}).get(key))
            if value and value not in ("none", "unknown", "unknownitem"):
                acc[value] = acc.get(value, 0.0) + chance
        return sorted(((k, v / total) for k, v in acc.items()), key=lambda kv: (-kv[1], kv[0])) if total > 0 else []

    def likely(self, key: str) -> List[Tuple[str, Optional[float]]]:
        """The opponent's active Pokemon's likely "item", "ability" or "tera_type": Foul Play's sampled sets, else its public sets."""
        got = self._weighted(key)
        if got:
            return [(k, p) for k, p in got]
        public = _set_of_species(getattr(self.theirs, "species", ""))
        names = public.get({"item": "items", "ability": "abilities", "tera_type": "tera"}[key]) or []
        return [(n, None) for n in names]

    def _maybe_moves(self) -> List[str]:
        """Moves the opponent's active Pokemon may also have (not shown yet): the sampled sets', else its public pool."""
        if self.theirs is None or len(self.foe_seen) >= 4:
            return []
        out: List[str] = []
        weights: Dict[str, float] = {}
        for w in self.worlds:
            for m in (w.get("active") or {}).get("moves") or []:
                weights[_id(m)] = weights.get(_id(m), 0.0) + (float(w.get("chance") or 0.0) or 1.0)
        pool = sorted(weights, key=lambda m: (-weights[m], m)) or list(_set_of_species(getattr(self.theirs, "species", "")).get("moves") or [])
        for m in pool:
            if m and m not in self.foe_seen and m not in out:
                out.append(m)
        return out

    # -- damage ------------------------------------------------------------------------------------

    def _screen(self, side: Dict[str, Any], category: str, attacker_ability: str) -> float:
        if attacker_ability == "infiltrator":
            return 1.0
        if "auroraveil" in side or (category == "PHYSICAL" and "reflect" in side) or (category == "SPECIAL" and "lightscreen" in side):
            return 0.5
        return 1.0

    def _entry(self, move: str, attacker: Any, tera: bool) -> Dict[str, Any]:
        """The move's data, with Tera Blast as it is once its user has Terastallized."""
        e = self.sm.move_entry(move)
        if _id(move) == "terablast" and attacker is not None and (tera or attacker.terastallized) and attacker.tera:
            atk = float(attacker.stats.get("atk") or 0) * self.sc.stage(int(attacker.boosts.get("atk", 0) or 0))
            spa = float(attacker.stats.get("spa") or 0) * self.sc.stage(int(attacker.boosts.get("spa", 0) or 0))
            e = dict(e, type=attacker.tera.capitalize(), category="Physical" if atk > spa else "Special")
        return e

    def our_damage(self, move: str, tera: bool, attacker: Any = None, hp_now: bool = True) -> Optional[Tuple[float, float]]:
        """(lowest, highest) damage roll of our move on the opponent's active Pokemon, as shares of its whole HP; None when unknown."""
        attacker = attacker if attacker is not None else self.our_f
        if attacker is None or self.foe_f is None:
            return None
        e = self._entry(move, attacker, tera)
        lo = self.sc.damage_share(e, attacker, self.foe_f, tera=tera, roll=0.85, weather=self.weather)
        hi = self.sc.damage_share(e, attacker, self.foe_f, tera=tera, roll=1.0, weather=self.weather)
        if lo is None or hi is None:
            return None
        k = self._screen(self.foe_side, str(e.get("category") or "").upper(), attacker.ability)
        if _id(move) == "knockoff" and self.foe_item_shown != "none":
            k *= 1.5          # Knock Off on a target that holds an item (damage_share does not count it); every random battle set has one
        return lo * k, hi * k

    def foe_hit(self, defender: Any, moves: Sequence[str]) -> Tuple[float, str]:
        """(the highest roll of the opponent's best of these moves on `defender`, as a share of its whole HP, the move)."""
        best, name = 0.0, ""
        if self.foe_f is None or defender is None:
            return best, name
        for m in moves:
            e = self.sm.move_entry(m)
            if str(e.get("category") or "") not in ("Physical", "Special"):
                continue
            share = self.sc.damage_share(e, self.foe_f, defender, roll=1.0, weather=self.weather)
            if share is None:
                continue
            share *= self._screen(self.our_side, str(e.get("category") or "").upper(), self.foe_f.ability)
            if _id(m) == "knockoff" and _id(getattr(defender, "item", "")) not in ("", "none", "unknownitem"):
                share *= 1.5      # Knock Off on our Pokemon while it holds its item
            if share > best:
                best, name = share, self.sm.move_name(m)
        return best, name

    def speeds(self, mon_f: Any = None) -> Optional[Tuple[float, float, Optional[float]]]:
        """(our speed, the opponent's estimated speed, the chance it holds a Choice Scarf or None when its item is known)."""
        mon_f = mon_f if mon_f is not None else self.our_f
        if mon_f is None or self.foe_f is None:
            return None
        ours, theirs = self.sc.speed(mon_f, self.weather), self.sc.speed(self.foe_f, self.weather)
        if "tailwind" in self.our_side:
            ours *= 2.0
        if "tailwind" in self.foe_side:
            theirs *= 2.0
        if ours <= 0 or theirs <= 0:
            return None
        scarf: Optional[float] = None
        if not self.foe_item:
            guesses = self.likely("item")
            if any(k == "choicescarf" for k, _p in guesses):
                p = next((p for k, p in guesses if k == "choicescarf"), None)
                scarf = float(p) if p is not None else 0.5
        return ours, theirs, scarf


def _mon_line(b: _Board, mon: Any, ours: bool) -> str:
    """ "Garchomp L77 (Dragon/Ground) HP 64%, burned, Atk+2" """
    sm = b.sm
    species = sm.species_name(str(getattr(mon, "species", "") or ""))
    types = "/".join(_cap(t) for t in (getattr(mon, "types", None) or []) if t is not None)
    text = "%s L%s (%s) HP %d%%" % (species, getattr(mon, "level", "?"), types or "?",
                                    max(0, round(100.0 * float(getattr(mon, "current_hp_fraction", 0.0) or 0.0))))
    status = _id(getattr(mon, "status", None))
    if status in _STATUS:
        text += ", " + _STATUS[status]
    boosts = " ".join("%s%+d" % (_BOOST.get(k, k), int(v)) for k, v in (getattr(mon, "boosts", None) or {}).items() if int(v or 0))
    if boosts:
        text += ", " + boosts
    effects = [_EFFECTS[_id(e)] for e in (getattr(mon, "effects", None) or {}) if _id(e) in _EFFECTS]
    if getattr(mon, "must_recharge", False):
        effects.append("must recharge")
    if effects:
        text += ", " + ", ".join(sorted(set(effects)))
    if getattr(mon, "is_terastallized", False):
        text += ", Terastallized"
    return text


def _listed(names: Sequence[Tuple[str, Optional[float]]], show: Callable[[str], str], most: int = 3) -> str:
    parts = []
    for name, p in list(names)[:most]:
        parts.append(show(name) + (" %s" % _pct(p) if p is not None and len(names) > 1 else ""))
    return ", ".join(parts)


def _who(ident: str, role: str) -> Tuple[str, str]:
    """ "p2a: Garchomp" -> ("our" | "foe's", "Garchomp") """
    side, _, name = str(ident).partition(":")
    return ("our" if side.strip()[:2] == role else "foe's"), name.strip()


def _hp_text(condition: str) -> str:
    """ "212/331 par" -> "64%"; "0 fnt" -> "0%" """
    first = str(condition or "").split(" ")[0]
    if "/" in first:
        try:
            num, den = first.split("/", 1)
            return "%d%%" % max(0, round(100.0 * float(num) / max(1.0, float(den))))
        except ValueError:
            return "?"
    return "0%" if first.startswith("0") else "?"


# The protocol lines kept of a turn ("what happened last turn" in the brief): what the Pokemon
# did. Never chat, a name a user typed, a raw HTML line or the timer.
EVENT_KINDS = frozenset((
    "move", "switch", "drag", "faint", "cant", "-damage", "-heal", "-status", "-curestatus", "-boost", "-unboost",
    "-clearallboost", "-clearboost", "-clearnegativeboost", "-clearpositiveboost", "-weather", "-fieldstart", "-fieldend",
    "-sidestart", "-sideend", "-terastallize", "-crit", "-immune", "-fail", "-miss", "-activate", "-singleturn", "-enditem",
    "-item", "-ability", "-start", "-end"))


class TurnLog:
    """
    What the Pokemon did in the turn being played (turn) and in the one before it (last), from a
    battle room's lines split at "|". A "|turn|" line closes the turn. Only EVENT_KINDS are kept.
    """

    def __init__(self, most: int = 120, keep: int = 4):
        self.turn: List[List[str]] = []
        self.last: List[List[str]] = []
        self.most = int(most)
        self.keep = int(keep)
        self.number = 0                                         # the turn being played (the last "|turn|N" seen)
        self.history: List[Tuple[int, List[List[str]]]] = []    # the last `keep` finished turns, oldest first: (turn, its lines)

    def feed(self, lines: Sequence[Sequence[str]]) -> None:
        for parts in lines:
            if len(parts) < 2 or parts[0] != "":
                continue
            if parts[1] == "turn":
                if self.turn:
                    self.history.append((self.number, self.turn))
                    del self.history[:-self.keep]
                self.last, self.turn = self.turn, []
                try:
                    self.number = int(parts[2])
                except (IndexError, ValueError):
                    self.number += 1
            elif parts[1] in EVENT_KINDS and len(self.turn) < self.most:
                self.turn.append(list(parts))

    def earlier(self) -> List[Tuple[int, List[List[str]]]]:
        """The finished turns before the last one (which is `last`), oldest first."""
        return list(self.history[:-1]) if self.history and self.history[-1][1] is self.last else list(self.history)


def render_events(lines: Sequence[Sequence[str]], role: str, most: int = 16, limit: int = 520) -> str:
    """
    A turn's protocol lines in a few plain words: who moved, who switched, the HP after a hit, what
    fainted, boosts, statuses, the field. Chat, names of users, timer lines and everything else a
    person could have typed are not read at all.
    """
    out: List[str] = []
    for parts in lines or ():
        try:
            if len(parts) < 3 or parts[0] != "":
                continue
            kind = parts[1]
            tail = [p for p in parts[2:]]
            source = next((p[len("[from]"):].strip() for p in tail if p.startswith("[from]")), "")
            source = re.sub(r"^(item|ability|move):\s*", "", source)
            if kind == "move" and len(parts) > 3:
                side, name = _who(parts[2], role)
                text = "%s %s used %s" % (side, name, parts[3])
                if any(p.startswith("[miss]") for p in tail):
                    text += " (missed)"
                out.append(text)
            elif kind in ("switch", "drag") and len(parts) > 3:
                side, name = _who(parts[2], role)
                out.append("%s %s %s" % (side, name, "came in" if kind == "switch" else "was dragged in"))
            elif kind == "faint":
                side, name = _who(parts[2], role)
                out.append("%s %s fainted" % (side, name))
            elif kind == "-damage" and len(parts) > 3:
                side, name = _who(parts[2], role)
                out.append("%s %s at %s%s" % (side, name, _hp_text(parts[3]), (" (%s)" % source) if source else ""))
            elif kind == "-heal" and len(parts) > 3:
                if source.lower() in ("leftovers", "black sludge", "grassy terrain", "aqua ring"):
                    continue
                side, name = _who(parts[2], role)
                out.append("%s %s healed to %s" % (side, name, _hp_text(parts[3])))
            elif kind == "-status" and len(parts) > 3:
                side, name = _who(parts[2], role)
                out.append("%s %s is %s" % (side, name, _STATUS.get(parts[3], parts[3])))
            elif kind == "-curestatus":
                side, name = _who(parts[2], role)
                out.append("%s %s was cured" % (side, name))
            elif kind in ("-boost", "-unboost") and len(parts) > 4:
                side, name = _who(parts[2], role)
                out.append("%s %s %s%s%s" % (side, name, _BOOST.get(parts[3], parts[3]), "+" if kind == "-boost" else "-", parts[4]))
            elif kind in ("-clearallboost", "-clearboost", "-clearnegativeboost", "-clearpositiveboost"):
                out.append("stat changes were cleared")
            elif kind == "-weather" and len(parts) > 2 and not any(p.startswith("[upkeep]") for p in tail):
                out.append("weather: %s" % (_WEATHER.get(_id(parts[2]), parts[2]) if _id(parts[2]) != "none" else "cleared"))
            elif kind in ("-fieldstart", "-fieldend") and len(parts) > 2:
                name = re.sub(r"^move:\s*", "", parts[2])
                out.append("%s %s" % (name, "started" if kind == "-fieldstart" else "ended"))
            elif kind in ("-sidestart", "-sideend") and len(parts) > 3:
                side = "our" if parts[2].strip()[:2] == role else "foe's"
                name = re.sub(r"^move:\s*", "", parts[3])
                out.append("%s on %s side%s" % (name, side, "" if kind == "-sidestart" else " ended"))
            elif kind == "-terastallize" and len(parts) > 3:
                side, name = _who(parts[2], role)
                out.append("%s %s Terastallized to %s" % (side, name, parts[3]))
            elif kind == "cant" and len(parts) > 3:
                side, name = _who(parts[2], role)
                out.append("%s %s could not move (%s)" % (side, name, re.sub(r"^(ability|move|item):\s*", "", parts[3])))
            elif kind == "-crit":
                out.append("a critical hit")
            elif kind == "-immune":
                side, name = _who(parts[2], role)
                out.append("%s %s was immune" % (side, name))
            elif kind == "-fail":
                out.append("it failed")
            elif kind == "-miss":
                out.append("it missed")
            elif kind in ("-activate", "-singleturn") and len(parts) > 3 and _id(re.sub(r"^move:\s*", "", parts[3])) in (
                    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap", "burningbulwark"):
                side, name = _who(parts[2], role)
                out.append("%s %s protected itself" % (side, name))
            elif kind == "-enditem" and len(parts) > 3:
                side, name = _who(parts[2], role)
                out.append("%s %s lost its %s" % (side, name, parts[3]))
            elif kind == "-item" and len(parts) > 3:
                side, name = _who(parts[2], role)
                out.append("%s %s holds %s" % (side, name, parts[3]))
            elif kind == "-ability" and len(parts) > 3:
                side, name = _who(parts[2], role)
                out.append("%s %s has %s" % (side, name, parts[3]))
            elif kind in ("-start", "-end") and len(parts) > 3:
                name = re.sub(r"^(move|ability|item):\s*", "", parts[3])
                if _id(name) in ("substitute", "leechseed", "confusion", "encore", "taunt", "yawn", "disable", "saltcure", "curse",
                                 "perish3", "perish2", "perish1", "protosynthesis", "quarkdrive", "futuresight", "doomdesire"):
                    side, who = _who(parts[2], role)
                    out.append("%s %s: %s %s" % (side, who, name, "began" if kind == "-start" else "ended"))
        except Exception:
            continue
    if len(out) > most:
        out = ["..."] + out[-most:]
    text = "; ".join(out)
    if len(text) > limit:
        text = "... " + text[-limit:].split("; ", 1)[-1]
    return text


def _move_effect(b: _Board, move: str, e: Dict[str, Any]) -> str:
    """What a move does besides damage, in a few words, from its data (and _MOVE_NOTES)."""
    notes: List[str] = []
    ident = _id(move)
    if ident in _MOVE_NOTES:
        notes.append(_MOVE_NOTES[ident])
    status_move = str(e.get("category") or "") == "Status"
    target_self = str(e.get("target") or "") in ("self", "adjacentAllyOrSelf", "allies", "allySide", "allyTeam")
    boosts = e.get("boosts") or {}
    if boosts and ident not in _MOVE_NOTES:
        words = " ".join("%s%+d" % (_BOOST.get(k, k), int(v)) for k, v in boosts.items())
        notes.append(("our " if target_self else "the target's ") + words)
    own = ((e.get("self") or {}).get("boosts") or {}) if isinstance(e.get("self"), dict) else {}
    if own and ident not in _MOVE_NOTES:
        notes.append("then our " + " ".join("%s%+d" % (_BOOST.get(k, k), int(v)) for k, v in own.items()))
    status = _id(e.get("status"))
    if status in _STATUS and ident not in _MOVE_NOTES:
        notes.append({"brn": "burns", "par": "paralyzes", "psn": "poisons", "tox": "badly poisons", "slp": "puts to sleep",
                      "frz": "freezes"}[status])
    secondary = e.get("secondary") if isinstance(e.get("secondary"), dict) else None
    if secondary and not status_move:
        chance = secondary.get("chance")
        what = _id(secondary.get("status")) or _id(secondary.get("volatileStatus"))
        word = {"brn": "burn", "par": "paralysis", "psn": "poison", "tox": "bad poison", "slp": "sleep", "frz": "freeze",
                "flinch": "flinch", "confusion": "confusion"}.get(what)
        if word and chance:
            notes.append("%s%% %s" % (chance, word))
        elif secondary.get("boosts") and chance:
            notes.append("%s%% the target's %s" % (chance, " ".join("%s%+d" % (_BOOST.get(k, k), int(v)) for k, v in secondary["boosts"].items())))
        elif isinstance(secondary.get("self"), dict) and (secondary["self"].get("boosts")) and chance and ident not in _MOVE_NOTES:
            notes.append("%s%% our %s" % (chance, " ".join("%s%+d" % (_BOOST.get(k, k), int(v)) for k, v in secondary["self"]["boosts"].items())))
    heal = e.get("heal")
    if isinstance(heal, (list, tuple)) and len(heal) == 2 and ident not in _MOVE_NOTES:
        notes.append("heals %d%% of our HP" % round(100.0 * heal[0] / max(1, heal[1])))
    drain = e.get("drain")
    if isinstance(drain, (list, tuple)) and len(drain) == 2:
        notes.append("heals us by %d%% of the damage" % round(100.0 * drain[0] / max(1, drain[1])))
    recoil = e.get("recoil")
    if isinstance(recoil, (list, tuple)) and len(recoil) == 2:
        notes.append("%d%% recoil" % round(100.0 * recoil[0] / max(1, recoil[1])))
    if e.get("selfSwitch") and ident not in _MOVE_NOTES:
        notes.append("we switch out after it")
    side = _id(e.get("sideCondition"))
    if side in _SIDE and ident not in _MOVE_NOTES:
        notes.append("sets %s" % _SIDE[side])
    if e.get("weather") and ident not in _MOVE_NOTES:
        notes.append("sets %s" % _WEATHER.get(_id(e.get("weather")), str(e.get("weather"))))
    if e.get("terrain"):
        notes.append("sets %s" % _FIELD.get(_id(e.get("terrain")), str(e.get("terrain"))))
    own_state = _id((e.get("self") or {}).get("volatileStatus")) if isinstance(e.get("self"), dict) else ""
    if own_state == "mustrecharge":
        notes.append("we must recharge next turn")
    elif own_state == "lockedmove":
        notes.append("locks us into it for 2-3 turns, then confuses us")
    multihit = e.get("multihit")
    if isinstance(multihit, (list, tuple)) and len(multihit) == 2:
        notes.append("hits %d-%d times" % (multihit[0], multihit[1]))
    elif isinstance(multihit, int) and multihit > 1:
        notes.append("hits %d times" % multihit)
    if "charge" in (e.get("flags") or {}) and ident not in _MOVE_NOTES:
        notes.append("charges a turn first")
    if ident in ("fakeout", "firstimpression") and b.ours is not None and not getattr(b.ours, "first_turn", False):
        notes.append("NOT our first turn in: it fails now")
    return "; ".join(notes)


def _eff_word(mult: float) -> str:
    if mult <= 0:
        return "no effect"
    if mult >= 4:
        return "4x super effective"
    if mult >= 2:
        return "super effective"
    if mult <= 0.25:
        return "doubly resisted"
    if mult < 1:
        return "resisted"
    return "neutral"


def _ko_text(b: _Board, lo: float, hi: float, accuracy: float, doubt: str) -> str:
    """ "KO chance about 85%" from the damage rolls, the opponent's HP now and the move's accuracy; or how many hits it takes."""
    hp = b.foe_hp
    if hp <= 0 or hi <= 0:
        return ""
    if lo >= hp:
        chance = accuracy
    elif hi >= hp:
        chance = accuracy * (hi - hp) / max(1e-9, hi - lo)
    else:
        chance = 0.0
    if chance > 0.004:
        return "KO chance about %s%s" % (_pct(chance), (" unless %s" % doubt) if doubt else "")
    hits = int(math.ceil(hp / max(1e-9, (lo + hi) / 2.0)))
    return "no KO this turn (about %d hits)" % hits if hits <= 6 else "no KO this turn"


def _behind_substitute(b: _Board, e: Dict[str, Any]) -> bool:
    """
    Whether the opponent's active Pokemon is behind a Substitute that takes this move's hit: not a
    sound move or another that goes through one, not from Infiltrator, and not a move of several
    hits (the hits after the one that breaks the Substitute land). As setup_counter.counter_check.
    """
    try:
        if not any(_id(x) == "substitute" for x in (getattr(b.theirs, "effects", None) or {})):
            return False
        flags = set((e.get("flags") or {}).keys())
        if flags & {"bypasssub", "sound"} or e.get("multihit"):
            return False
        return _id(getattr(b.our_f, "ability", "")) != "infiltrator"
    except Exception:
        return False


def _doubts(b: _Board, e: Dict[str, Any]) -> Tuple[str, str]:
    """
    (an ability the opponent's active Pokemon may have that takes or weakens this move, what could
    let it live through any one hit at full HP): each "" when there is no such doubt. A known
    ability is not a doubt: the damage estimate already counts it.
    """
    ability = survive = ""
    try:
        import search_player as sp

        if not b.foe_ability and b.theirs is not None:
            move_type, category = str(e.get("type") or "").upper(), str(e.get("category") or "").upper()
            pool = list(_set_of_species(getattr(b.theirs, "species", "")).get("abilities") or []) or [
                _id(a) for a in (getattr(b.theirs, "possible_abilities", None) or ()) if a]
            for ab in pool:
                types = sp._KO_ABILITIES.get(ab)
                if (types and ("*" in types or move_type in types)) or sp._KO_CATEGORY_ABILITIES.get(ab) == category:
                    ability = _ability_name(ab)
                    break
        if b.foe_hp >= 1.0 and b.theirs is not None:
            guard = b.sc.full_hp_guard(b.theirs, b.worlds)
            if guard and guard != "unknown":
                survive = {"focussash": "its Focus Sash", "focussash?": "a Focus Sash"}.get(guard, "its %s" % _ability_name(guard))
    except Exception:
        pass
    return ability, survive


def _advisors(b: _Board, a: int, q: Sequence[float]) -> str:
    f = b.facts
    if getattr(f, "solo", False):
        return ""                       # the model plays alone: no network, no search, no pick of theirs
    text = "net %s" % _pct(q[a])
    if f.shares is not None:
        text += ", search %s" % _pct(f.shares[a])
        score = f.scores[a] if f.scores is not None else None
        if score is not None:
            text += " (result %.2f)" % float(score)
    row = f.lines.get(a) if f.lines else None
    if row is not None and row.get("value") is not None:
        text += ", line %.2f after %d turns" % (float(row["value"]), int(row.get("turns") or 0))
    tags = []
    if a == f.mix_action:
        tags.append("MIX'S PICK")
    return text + ("." if not tags else ". " + ", ".join(tags) + ".")


def _move_choice(b: _Board, a: int, move: Any, tera: bool, q: Sequence[float], detail: int) -> Tuple[str, str]:
    sm, sc = b.sm, b.sc
    ident = _id(getattr(move, "id", move))
    e = b._entry(ident, b.our_f, tera) if b.our_f is not None else sm.move_entry(ident)
    name = sm.move_name(ident)
    category = str(e.get("category") or "Status")
    kind = _cap(e.get("type") or "?")
    head = name
    tera_type = b.own_tera(b.ours) if b.ours is not None else ""
    if tera:
        head += " + Terastallize" + (" (we become pure %s)" % tera_type if tera_type else "")
    facts = "%s %s" % (kind, category.lower())
    priority = int(e.get("priority") or 0)
    if category != "Status":
        power = e.get("basePower") or 0
        accuracy = e.get("accuracy", True)
        facts += ", power %s" % (power if power else "varies")
        facts += ", accuracy %s" % ("sure" if accuracy is True else "%s%%" % accuracy)
    else:
        accuracy = e.get("accuracy", True)
        if accuracy is not True:
            facts += ", accuracy %s%%" % accuracy
    if priority:
        facts += ", priority %+d" % priority
    effect = _move_effect(b, ident, e) if detail < 3 else ""
    text = "%s: %s" % (head, facts) + ("; " + effect if effect else "") + "."
    if category != "Status" and b.theirs is not None:
        mult = None
        try:
            mult = sc.effectiveness(str(e.get("type") or ""), sc._types(getattr(b.theirs, "types", ())))
        except Exception:
            pass
        rolls = b.our_damage(ident, tera)
        maybe_ability, survive = _doubts(b, e)
        if rolls is not None and rolls[1] <= 0:
            reason = "immune by type" if (mult is not None and mult <= 0) else (
                "its ability %s%s" % (_ability_name(b.foe_ability), "" if b.foe_ability_shown else ", which its sets all have")
                if b.foe_ability else "immune")
            text += " Does NOTHING to the foe's active Pokemon (%s)." % reason
        elif rolls is not None:
            lo, hi = rolls
            acc = 1.0 if accuracy is True or not isinstance(accuracy, (int, float)) else max(0.0, min(1.0, float(accuracy) / 100.0))
            ko = _ko_text(b, lo, hi, acc, survive)
            if ko.startswith("KO chance") and _behind_substitute(b, e):
                ko = "no KO this turn: its Substitute takes this hit"
            text += " Est. %d-%d%% of the foe's HP" % (round(100 * lo), round(100 * hi))
            if mult is not None and mult != 1.0:
                text += " (%s)" % _eff_word(mult)
            text += ("; " + ko if ko else "") + "."
            if maybe_ability:
                text += " Less or nothing if it has %s (possible)." % maybe_ability
        elif mult is not None and mult <= 0:
            text += " Does NOTHING to the foe's active Pokemon (immune by type)."
        else:
            text += " Damage not estimated" + (" (%s)" % _eff_word(mult) if mult is not None and mult != 1.0 else "") + "."
            if maybe_ability:
                text += " Less or nothing if it has %s (possible)." % maybe_ability
    if tera and b.our_f is not None and tera_type and detail < 2:
        # What Terastallizing does to the hits we take: our typing changes with it.
        try:
            after = replace(b.our_f, types=(tera_type.upper(),), terastallized=True)
            moves = b.foe_seen or b.foe_maybe[:6]
            now, _n = b.foe_hit(b.our_f, moves)
            then, then_name = b.foe_hit(after, moves)
            if moves and abs(now - then) >= 0.05:
                text += " As %s we take up to %d%% from its %s moves (now %d%%)." % (
                    tera_type, round(100 * then), "seen" if b.foe_seen else "likely", round(100 * now))
        except Exception:
            pass
    text += " " + _advisors(b, a, q)
    value = "a%d-%s%s" % (a, ident[:24] or "move", "-tera" if tera else "")
    return value, text


def _hazards_on_entry(b: _Board, mon: Any, f: Any) -> str:
    side = b.our_side
    if not any(k in side for k in ("stealthrock", "spikes", "toxicspikes", "stickyweb")):
        return ""
    if _id(getattr(mon, "item", "")) == "heavydutyboots":
        return "hazards do nothing to it (Heavy-Duty Boots)"
    ability = _id(getattr(mon, "ability", ""))
    types = [str(t).upper() for t in f.types]
    grounded = "FLYING" not in types and ability != "levitate" and _id(getattr(mon, "item", "")) != "airballoon"
    parts: List[str] = []
    loss = 0.0
    if "stealthrock" in side and ability != "magicguard":
        loss += 0.125 * b.sc.effectiveness("ROCK", types)
    if "spikes" in side and grounded and ability != "magicguard":
        loss += {1: 0.125, 2: 1.0 / 6.0}.get(int(side.get("spikes") or 1), 0.25)
    if loss > 0:
        parts.append("loses %d%% to hazards on entry" % round(100 * loss))
    if "toxicspikes" in side and grounded:
        if "POISON" in types:
            parts.append("absorbs the Toxic Spikes")
        elif "STEEL" not in types and not _id(getattr(mon, "status", None)):
            parts.append("is poisoned by Toxic Spikes")
    if "stickyweb" in side and grounded:
        parts.append("Spe -1 from Sticky Web")
    return ", ".join(parts)


def _switch_choice(b: _Board, a: int, mon: Any, q: Sequence[float], detail: int) -> Tuple[str, str]:
    sm = b.sm
    f = b._try(lambda: b.fighter(mon), None)
    text = "Switch to " + _mon_line(b, mon, True)
    item = _id(getattr(mon, "item", ""))
    ability = _id(getattr(mon, "ability", ""))
    extras = []
    if item and item not in ("unknownitem",):
        extras.append(_item_name(item))
    if ability:
        extras.append(_ability_name(ability))
    if extras and detail < 3:
        text += "; " + ", ".join(extras)
    moves = [_id(m) for m in (getattr(mon, "moves", None) or {})]
    if moves and detail < 2:
        text += "; moves " + ", ".join(sm.move_name(m) for m in moves[:4])
    text += "."
    if f is not None and b.theirs is not None:
        if b.foe_seen:
            hit, name = b.foe_hit(f, b.foe_seen)
            if name:
                text += " Takes up to %d%% from the foe's seen moves (%s)." % (round(100 * hit), name)
            else:
                text += " The foe's seen moves do no damage to it."
        maybe = [m for m in b.foe_maybe[:8] if m not in b.foe_seen]
        if maybe and detail < 2:
            hit2, name2 = b.foe_hit(f, maybe)
            if name2 and (not b.foe_seen or hit2 > 1.15 * b.foe_hit(f, b.foe_seen)[0]):
                text += " A move it may have (%s) could do %d%%." % (name2, round(100 * hit2))
        best, best_name = 0.0, ""
        for m in moves:
            rolls = b.our_damage(m, False, attacker=f)
            if rolls is not None and rolls[1] > best:
                best, best_name = rolls[1], sm.move_name(m)
        if best_name:
            text += " Its best hit on the foe: %s, up to %d%%." % (best_name, round(100 * best))
        speeds = b.speeds(f)
        if speeds is not None and detail < 3:
            ours, theirs, _scarf = speeds
            faster = (ours < theirs) if b.trick_room else (ours > theirs)
            close = abs(ours - theirs) <= 0.04 * max(ours, theirs)
            text += " About as fast as the foe." if close else " %s than the foe." % ("Faster" if faster else "Slower")
        hazards = _hazards_on_entry(b, mon, f)
        if hazards:
            text += " It %s." % hazards if not hazards.startswith("hazards") else " Our side's %s." % hazards
    if getattr(b.facts, "solo", False) and not b.forced:
        text += " COST: it uses our turn, and the foe acts freely against it as it comes in."
    text += " " + _advisors(b, a, q)
    value = "a%d-switch-%s" % (a, _id(getattr(mon, "species", ""))[:22] or "pokemon")
    return value, text


def _foe_next(b: _Board) -> str:
    """The opponent's likely next action: our model's read, else the search's own shares, in plain names."""
    import search_player as sp

    f = b.facts
    types = [str(getattr(t, "name", t)) for t in (getattr(b.theirs, "types", None) or ()) if t is not None]
    rows: List[Tuple[str, float]] = []
    source = "our model of opponents"
    if f.prediction:
        rows = [(str(n), float(p)) for n, p in f.prediction]
    elif f.opponent_options:
        source = "the search's own read"
        total = sum(max(0.0, float(s)) for _n, s, _seen in f.opponent_options) or 1.0
        folded: Dict[str, float] = {}
        for name, share, seen in f.opponent_options:
            key = "switch ?" if str(name).startswith("switch ") and seen is False else str(name)
            folded[key] = folded.get(key, 0.0) + max(0.0, float(share)) / total
        rows = sorted(folded.items(), key=lambda kv: (-kv[1], kv[0]))
    if not rows:
        return ""
    shown = [(n, p) for n, p in rows[:5] if p >= 0.04] or rows[:1]
    rest = max(0.0, 1.0 - sum(p for _n, p in shown))
    text = ", ".join("%s %s" % (sp.option_text(n, types), _pct(p)) for n, p in shown)
    if rest >= 0.05:
        text += ", something else %s" % _pct(rest)
    return "FOE'S LIKELY NEXT ACTION (%s): %s." % (source, text)


def _last_move_of(lines: Sequence[Sequence[str]], role: str, ours: bool) -> str:
    """The id of the last move one side used in these lines ("" when none)."""
    last = ""
    for parts in lines or ():
        if len(parts) > 3 and parts[0] == "" and parts[1] == "move" and (parts[2].strip()[:2] == role) == ours:
            last = _id(parts[3])
    return last


def _threats(b: _Board, q: Sequence[float]) -> str:
    """Set-up and sweep threats, and whether we have an answer on the field."""
    sm, sc = b.sm, b.sc
    f = b.facts
    notes: List[str] = []
    theirs, ours = b.theirs, b.ours
    if theirs is None:
        return ""
    name = sm.species_name(str(getattr(theirs, "species", "") or ""))
    types = [str(getattr(t, "name", t)) for t in (getattr(theirs, "types", None) or ()) if t is not None]
    boosts = {k: int(v) for k, v in (getattr(theirs, "boosts", None) or {}).items() if int(v or 0) > 0 and k != "accuracy"}
    last = _last_move_of(f.this_turn, b.role, False) or _last_move_of(f.last_turn, b.role, False)
    last_setup = bool(last and sm.is_setup(last, types))
    if boosts:
        notes.append("the foe's %s is boosted (%s)%s" % (name, " ".join("%s%+d" % (_BOOST.get(k, k), v) for k, v in boosts.items()),
                                                        ", it set up with %s last" % sm.move_name(last) if last_setup else ""))
    shown = [m for m in b.foe_seen if sm.is_setup(m, types)]
    maybe = [m for m in b.foe_maybe[:8] if sm.is_setup(m, types)]
    if shown:
        notes.append("it has shown %s" % ", ".join(sm.move_name(m) for m in shown))
    elif maybe:
        notes.append("it may carry %s" % ", ".join(sm.move_name(m) for m in maybe[:2]))
    read = sleep = 0.0
    for move_name, p in (f.prediction or []):
        if str(move_name).startswith("move "):
            ident = str(move_name)[5:].replace(" terastallize", "")
            if sm.is_setup(ident, types):
                read += float(p)
            if sm.is_sleep(ident):
                sleep += float(p)
    if read >= 0.25:
        notes.append("our model gives %s that it sets up this turn" % _pct(read))
    if sleep >= 0.25 and ours is not None and getattr(ours, "status", None) is None:
        notes.append("our model gives %s that it uses a sleep move" % _pct(sleep))
    if ours is not None and b.our_f is not None and not b.forced:
        our_hp = float(getattr(ours, "current_hp_fraction", 0.0) or 0.0)
        hit, hit_name = b.foe_hit(b.our_f, b.foe_seen)
        if hit_name:
            text = "its seen moves do up to %d%% to our active (%s)" % (round(100 * hit), hit_name)
            if hit >= our_hp > 0:
                text += ": that can knock it out (it has %d%% left)" % round(100 * our_hp)
            notes.append(text)
        maybe_moves = [m for m in b.foe_maybe[:8] if m not in b.foe_seen]
        if maybe_moves:
            hit2, name2 = b.foe_hit(b.our_f, maybe_moves)
            if name2 and hit2 > max(hit * 1.15, 0.2):
                notes.append("a move it may have (%s) could do %d%%" % (name2, round(100 * hit2)))
        try:
            board = sc.build_board(b.battle, {a: v for a, v in f.actions.items()},
                                   threat=SimpleNamespace(last_setup=last_setup, setup_read=read), advice=SimpleNamespace(opponent_worlds=b.worlds))
            if boosts or last_setup or read >= 0.25:
                works = [str(v).split(" ", 1)[-1] for _a, v in sorted(board.counters.items())]
                if works:
                    notes.append("our answer on the field: %s" % ", ".join(sm.move_name(m) for m in works))
                failed = [str(v) for _a, v in sorted(board.failed.items())]
                if failed:
                    notes.append("not working now: %s" % "; ".join(failed[:2]))
        except Exception:
            pass
        ours_boosts = {k: int(v) for k, v in (getattr(ours, "boosts", None) or {}).items() if int(v or 0) > 0}
        if ours_boosts:
            notes.append("our active is boosted (%s)" % " ".join("%s%+d" % (_BOOST.get(k, k), v) for k, v in ours_boosts.items()))
    if not notes:
        return ""
    return "THREATS: " + "; ".join(notes) + "."


def _line_side(b: _Board, side: Dict[str, Any], ours: bool) -> str:
    """ "our Garchomp 41% (burned, +2 Atk)" from a line's snapshot of a side. """
    sm = b.sm
    name = _id(side.get("active"))
    who = "our " if ours else "foe's "
    text = who + sm.species_name(name)
    if not ours and b.facts.revealed and name not in [_id(n) for n in b.facts.revealed]:
        text += " (not shown yet: a guess)"
    hp = int(side.get("hp") or 0)
    if hp <= 0:
        return text + " fainted"
    text += " %d%%" % hp
    more = []
    if side.get("status"):
        more.append(_STATUS.get(str(side["status"]), str(side["status"])))
    for stat, stage in sorted((side.get("boosts") or {}).items()):
        more.append("%+d %s" % (int(stage), _BOOST.get(stat, stat)))
    return text + (" (%s)" % ", ".join(more) if more else "")


def _line_action(b: _Board, name: str, we: bool = False) -> str:
    """ "move earthquake terastallize" / "switch garchomp" / "none" of a line's step, in words: "uses Earthquake and Terastallizes" (we: "use ... and Terastallize"). """
    sm = b.sm
    name = str(name or "none")
    s = "" if we else "s"
    if name.startswith("switch "):
        return ("switch%s to " % ("" if we else "es")) + sm.species_name(_id(name[len("switch "):]))
    if name.startswith("move "):
        words = name[len("move "):].split(" ")
        tera = len(words) > 1 and words[-1] == "terastallize"
        return "use%s " % s + sm.move_name(_id(words[0])) + ((" and Terastallize%s" % s) if tera else "")
    return "do%s nothing" % ("" if we else "es")


def _look_ahead(b: _Board, detail: int) -> str:
    """The LOOK-AHEAD LINES of a lines arm: per action that has one, its steps in the likeliest world read and how it ends."""
    f = b.facts
    if not f.lines or detail >= 3:
        return ""
    out: List[str] = []
    turns = max(int(row.get("turns") or 0) for row in f.lines.values())
    out.append("LOOK-AHEAD LINES (our search played each of its favourite actions forward, up to %d turns: both sides choose by "
               "search, every turn takes its likeliest outcome, in the foe's likeliest sampled set; value = the search's "
               "result at the end, 0 lost to 1 won):" % turns)
    most = 6 if detail == 0 else 3 if detail == 1 else 0
    for a, row in sorted(f.lines.items(), key=lambda kv: -(kv[1].get("value") or 0.0)):
        if row.get("value") is None or a not in f.actions:
            continue
        worlds = row.get("worlds") or []
        target, kw = f.actions[a]
        if hasattr(target, "current_hp"):
            label = "Switch to " + b.sm.species_name(str(getattr(target, "species", "")))
        else:
            label = b.sm.move_name(_id(getattr(target, "id", ""))) + (" + Terastallize" if (kw or {}).get("terastallize") else "")
        text = "- %s: " % label
        steps = list((worlds[0].get("steps") if worlds else None) or [])
        told: List[str] = []
        turn = 0
        for step in steps[:most + 2 if most else 0]:
            after = step.get("after") or {}
            if step.get("forced"):
                head = "then"
            else:
                turn += 1
                if turn > most:
                    break
                head = "T%d" % turn
            acts = []
            if str(step.get("ours") or "none") != "none":
                acts.append("we " + _line_action(b, step["ours"], we=True))
            if str(step.get("theirs") or "none") != "none":
                acts.append("foe " + _line_action(b, step["theirs"]))
            sides = "%s, %s" % (_line_side(b, after.get("ours") or {}, True), _line_side(b, after.get("theirs") or {}, False))
            told.append("%s %s -> %s" % (head, " and ".join(acts) or "nothing happens", sides))
        if told:
            text += ". ".join(told) + ". "
        last = (steps[-1].get("after") if steps else None) or {}
        if last:
            text += "END after %d turns: we have %d left (%s of our team's HP), foe %d (%s). " % (
                int(row.get("turns") or 0), int((last.get("ours") or {}).get("left") or 0), _pct((last.get("ours") or {}).get("team_hp")),
                int((last.get("theirs") or {}).get("left") or 0), _pct((last.get("theirs") or {}).get("team_hp")))
        text += "Value %.2f" % float(row["value"])
        if len(worlds) > 1:
            text += " (%s in the sets read)" % ", ".join("%.2f" % float(w["value"]) for w in worlds)
        out.append(text + ".")
    return "\n".join(out) if len(out) > 1 else ""


def _foe_move_notes(b: _Board, moves: Sequence[str], types: Sequence[str], defender: Any) -> List[str]:
    """ "Earthquake 73%" / "Calm Mind (sets up)" / "Thunder Wave (status)" for each of the opponent's moves, the hardest hits first. """
    sm = b.sm
    hits: List[Tuple[float, str]] = []
    others: List[str] = []
    for m in moves:
        e = sm.move_entry(m)
        name = sm.move_name(m)
        if str(e.get("category") or "") in ("Physical", "Special"):
            first = " (moves first)" if int(e.get("priority") or 0) > 0 else ""
            if defender is None:            # nobody of ours on the field (a forced switch): the move, without a number
                hits.append((float(e.get("basePower") or 0.0), name + first))
            else:
                share, _n = b.foe_hit(defender, [m])
                hits.append((share, "%s %d%%%s" % (name, round(100 * share), first)))
        else:
            others.append("%s (%s)" % (name, "sets up" if sm.is_setup(m, types) else "puts to sleep" if sm.is_sleep(m) else "status"))
    return [text for _share, text in sorted(hits, key=lambda h: -h[0])] + others


def _foe_options(b: _Board, detail: int) -> str:
    """
    A model alone: what the opponent's active Pokemon can carry (the sets random battles give it;
    Foul Play's sampled sets are not there without a search) and what each move does to our active
    Pokemon, then the same pool for each of its seen bench Pokemon with its possible items.
    """
    sm = b.sm
    if b.theirs is None or detail >= 3:
        return ""
    types = [str(getattr(t, "name", t)) for t in (getattr(b.theirs, "types", None) or ()) if t is not None]
    name = sm.species_name(str(getattr(b.theirs, "species", "") or ""))
    # Against our active Pokemon; on a forced switch nobody is there, and each switch choice says what it takes.
    target = None if b.forced else b.our_f
    shown = _foe_move_notes(b, b.foe_seen, types, target)
    maybe = _foe_move_notes(b, [m for m in b.foe_maybe[:10 if detail < 1 else 6] if m not in b.foe_seen], types, target)
    text = "FOE'S POSSIBLE MOVES (%s; the sets random battles give it%s): shown: %s" % (
        name, "; damage as a share of our active's whole HP" if target is not None else "; what each does to a Pokemon coming in is in its switch choice",
        ", ".join(shown) if shown else "none yet")
    if maybe:
        text += "; may also have%s: %s" % ("" if len(b.foe_seen) < 3 else " (one more)", ", ".join(maybe))
    elif len(b.foe_seen) >= 4:
        text += "; all four of its moves are shown"
    text += "."
    if detail < 2:
        rows: List[str] = []
        for mon in (getattr(b.battle, "opponent_team", None) or {}).values():
            if getattr(mon, "active", False) or getattr(mon, "fainted", False):
                continue
            pool = _set_of_species(getattr(mon, "species", ""))
            seen = [_id(m) for m in (getattr(mon, "moves", None) or {})]
            more = [m for m in pool.get("moves") or [] if m not in seen][:6 if detail < 1 else 4]
            row = sm.species_name(str(getattr(mon, "species", "") or ""))
            parts = []
            if more and len(seen) < 4:
                parts.append("may also have " + ", ".join(sm.move_name(m) for m in more))
            item = _id(getattr(mon, "item", ""))
            if (not item or item == "unknownitem") and pool.get("items"):
                parts.append("item one of " + ", ".join(_item_name(i) for i in pool["items"][:3]))
            if pool.get("abilities") and not _id(getattr(mon, "ability", "")):
                parts.append("ability one of " + ", ".join(_ability_name(x) for x in pool["abilities"][:2]))
            if parts:
                rows.append(row + ": " + "; ".join(parts))
        if rows:
            text += " ITS BENCH, seen so far: " + ". ".join(rows) + "."
    return text


def _memory(b: _Board, detail: int) -> str:
    """A model alone: its own last moves in this battle with their reasons, and how often it has been switching."""
    f = b.facts
    rows = list(f.memory or [])[-(5 if detail < 2 else 3):]
    if not rows:
        return ""
    told = []
    for row in rows:
        text = "T%s %s" % (row.get("turn"), row.get("text") or "?")
        if row.get("forced"):
            text += " (forced)"
        if row.get("by") == "rule":
            text += " (no answer from you: a fallback rule chose)"
        elif row.get("reason") and detail < 1:
            text += " [your reason: %s]" % _one_line(row["reason"], 150)
        told.append(text)
    free = [row for row in rows if not row.get("forced")]
    switches = sum(1 for row in free if row.get("switch"))
    text = "YOUR LAST MOVES (oldest first): " + "; ".join(told) + "."
    if free:
        text += " Of your last %d free moves, %d were switches." % (len(free), switches)
        if switches >= 2 and switches * 2 >= len(free):
            text += " That is a lot: each one gave the foe a free turn. Switch again only if staying in loses the Pokemon for nothing."
    return text


def _earlier(b: _Board, detail: int) -> str:
    """A model alone: what happened on the turns before the last one, briefly."""
    f = b.facts
    if detail >= 2 or not f.earlier:
        return ""
    told = []
    for turn, lines in list(f.earlier)[-(3 if detail < 1 else 2):]:
        text = render_events(lines, b.role, most=8, limit=240)
        if text:
            told.append("T%s: %s" % (turn, text))
    return ("EARLIER TURNS: " + ". ".join(told) + ".") if told else ""


def _brief(b: _Board, q: Sequence[float], detail: int) -> str:
    sm = b.sm
    f, battle = b.facts, b.battle
    lines: List[str] = []

    def section(name: str, fn: Callable[[], str]) -> None:
        try:
            text = fn()
            if text:
                lines.append(text)
        except Exception:
            if name not in b.problems:
                b.problems.append(name)

    def head() -> str:
        text = "Pokemon Showdown, gen 9 random battle, singles. \"We\" are the side to act; \"foe\" is the opponent. Turn %d." % b.turn
        if b.forced:
            fainted = b.ours is None or bool(getattr(b.ours, "fainted", False)) or not float(getattr(b.ours, "current_hp_fraction", 0.0) or 0.0)
            text += (" REQUEST: forced switch. Our active Pokemon fainted: choose who comes in." if fainted else
                     " REQUEST: forced switch. Our active Pokemon leaves the field (a pivot move or a forced swap): choose who comes in.")
        else:
            text += " REQUEST: choose a move or a switch."
            tera = b.own_tera(b.ours) if b.ours is not None else ""
            if any(f.mask[a] > 0.5 for a in range(10, 14)):
                text += " We can Terastallize this turn" + (" (our active's Tera type: %s)." % tera if tera else ".")
            else:
                text += " We cannot Terastallize now."
            if not any(f.mask[a] > 0.5 for a in range(4, 10)) and any(
                    not getattr(m, "fainted", False) and not getattr(m, "active", False) for m in (getattr(battle, "team", None) or {}).values()):
                text += " We cannot switch (trapped)."
        return text

    def counts() -> str:
        team = list((getattr(battle, "team", None) or {}).values())
        foes = list((getattr(battle, "opponent_team", None) or {}).values())
        sizes = getattr(battle, "_team_size", None) or {}
        foe_role = "p2" if b.role == "p1" else "p1"
        ours_n = int(sizes.get(b.role) or len(team) or 6)
        foe_n = int(sizes.get(foe_role) or 6)
        unseen = max(0, foe_n - len(foes))
        text = "POKEMON LEFT: ours %d of %d; the foe's %d of %d" % (
            sum(1 for m in team if not getattr(m, "fainted", False)), ours_n,
            sum(1 for m in foes if not getattr(m, "fainted", False)) + unseen, foe_n)
        text += " (%d not seen yet)." % unseen if unseen else "."
        if f.win_est is not None and not f.solo:      # (a model alone is told nothing of the network's)
            text += " Our network's win estimate: %s." % _pct(f.win_est)
        return text

    def our_active() -> str:
        if b.ours is None or b.forced and getattr(b.ours, "fainted", False):
            return ""
        mon = b.ours
        text = "OUR ACTIVE: " + _mon_line(b, mon, True)
        item, ability = _id(getattr(mon, "item", "")), _id(getattr(mon, "ability", ""))
        text += "; item %s" % (_item_name(item) if item and item != "unknownitem" else "none")
        if ability:
            text += "; ability %s" % _ability_name(ability)
        tera = b.own_tera(mon)
        if tera and not getattr(mon, "is_terastallized", False):
            text += "; Tera type %s" % tera
        return text + "."

    def foe_active() -> str:
        mon = b.theirs
        if mon is None:
            return ""
        text = "FOE'S ACTIVE: " + _mon_line(b, mon, False)
        if b.foe_item_shown == "none":
            text += "; no item"
        elif b.foe_item_shown:
            text += "; item %s" % _item_name(b.foe_item_shown)
        elif b.foe_item and b.foe_item != "none":
            text += "; item not shown (its sets all have %s)" % _item_name(b.foe_item)
        else:
            guess = [(k, p) for k, p in b.likely("item")][:3]
            text += "; item not shown" + (" (likely %s)" % _listed(guess, _item_name) if guess else "")
        if b.foe_ability_shown:
            text += "; ability %s" % _ability_name(b.foe_ability_shown)
        elif b.foe_ability:
            text += "; ability not shown (its sets all have %s)" % _ability_name(b.foe_ability)
        else:
            guess = b.likely("ability")[:3]
            text += "; ability not shown" + (" (possible %s)" % _listed(guess, _ability_name) if guess else "")
        text += "; moves seen: %s" % (", ".join(sm.move_name(m) for m in b.foe_seen) if b.foe_seen else "none yet")
        if b.foe_maybe and detail < 3:
            text += "; may also have: %s" % ", ".join(sm.move_name(m) for m in b.foe_maybe[:6 if detail < 1 else 4])
        if not getattr(mon, "is_terastallized", False):
            used = bool(getattr(battle, "opponent_used_tera", False)) or any(
                getattr(m, "is_terastallized", False) for m in (getattr(battle, "opponent_team", None) or {}).values())
            if used:
                text += "; the foe has used its Tera"
            elif detail < 2:
                guess = b.likely("tera_type")[:3]
                text += "; can still Terastallize" + (" (possible types: %s)" % _listed(guess, _cap) if guess else "")
        return text + "."

    def speed() -> str:
        if b.forced:
            return ""
        got = b.speeds()
        if got is None:
            return ""
        ours, theirs, scarf = got
        if b.trick_room:
            first = ours < theirs
        else:
            first = ours > theirs
        if abs(ours - theirs) <= 0.04 * max(ours, theirs):
            # The opponent's speed is an estimate (its set's spread is not known to the point): this close is a coin flip.
            text = "SPEED: too close to call (ours %d, the foe's about %d)" % (round(ours), round(theirs))
        else:
            text = "SPEED: %s (ours %d, the foe's about %d%s)" % (
                "we move first" if first else "the foe moves first", round(ours), round(theirs),
                ", Trick Room is up" if b.trick_room else "")
        doubts = []
        if scarf is not None and first and not b.trick_room and theirs * 1.5 > ours:
            doubts.append("a Choice Scarf (%s of its sets) would make it faster" % _pct(scarf))
        priority = [sm.move_name(m) for m in b.foe_seen if int(sm.move_entry(m).get("priority") or 0) > 0
                    and str(sm.move_entry(m).get("category") or "") != "Status"]
        if priority:
            doubts.append("it has priority: %s" % ", ".join(priority))
        return text + ("; " + "; ".join(doubts) if doubts else "") + "."

    def bench(ours: bool) -> str:
        team = list((getattr(battle, "team" if ours else "opponent_team", None) or {}).values())
        alive, gone = [], []
        offered = {id(v[0]) for v in f.actions.values()}
        for mon in team:
            if getattr(mon, "active", False):
                continue
            name = sm.species_name(str(getattr(mon, "species", "") or ""))
            if getattr(mon, "fainted", False):
                gone.append(name)
                continue
            if ours and id(mon) in offered:
                continue                     # described in its switch choice
            row = _mon_line(b, mon, ours)
            if not ours:
                seen = [sm.move_name(_id(m)) for m in (getattr(mon, "moves", None) or {})]
                if seen and detail < 2:
                    row += ", seen: " + ", ".join(seen)
                item = _id(getattr(mon, "item", ""))
                if item and item != "unknownitem" and detail < 2:
                    row += ", " + _item_name(item)
            alive.append(row)
        label = "OUR BENCH" if ours else "FOE'S BENCH (seen so far)"
        if ours:
            if not alive and not gone:
                return ""
            text = label + ": "
            switches = sum(1 for a in f.actions if 4 <= a <= 9)
            text += ("%d can come in (see the switch choices)" % switches) if switches else "nobody can come in now"
            if alive:
                text += "; not available now: " + "; ".join(alive)
        else:
            text = label + ": " + ("; ".join(alive) if alive else "none alive seen")
        if gone:
            text += ". Fainted: " + ", ".join(gone)
        return text + "."

    def field_line() -> str:
        parts = []
        weather = [_WEATHER.get(_id(w), _cap(w)) for w in (getattr(battle, "weather", None) or {})]
        parts.append("weather %s" % (", ".join(weather) if weather else "none"))
        fields = [_FIELD.get(_id(x), _cap(x)) for x in (getattr(battle, "fields", None) or {})]
        if fields:
            parts.append(", ".join(fields))

        def side(conds: Dict[str, Any]) -> str:
            words = []
            for key, value in conds.items():
                if key not in _SIDE:
                    continue
                layers = " x%d" % int(value) if key in ("spikes", "toxicspikes") and isinstance(value, int) and value > 1 else ""
                words.append(_SIDE[key] + layers)
            return ", ".join(words) if words else "nothing"

        parts.append("our side: %s" % side(b.our_side))
        parts.append("foe's side: %s" % side(b.foe_side))
        return "FIELD: " + "; ".join(parts) + "."

    def events() -> str:
        if f.this_turn:
            text = render_events(f.this_turn, b.role)
            if text:
                return "THIS TURN SO FAR: " + text + "."
        text = render_events(f.last_turn, b.role)
        return ("LAST TURN: " + text + ".") if text else ""

    def advisors() -> str:
        import search_player as sp  # noqa: F401  (kept for the names it loads)

        def label(a: int) -> str:
            target, kw = f.actions.get(a, (None, {}))
            if target is None:
                return "action %d" % a
            if hasattr(target, "current_hp"):
                return "switch to " + sm.species_name(str(getattr(target, "species", "")))
            return sm.move_name(_id(getattr(target, "id", ""))) + (" + Terastallize" if (kw or {}).get("terastallize") else "")

        text = "ADVISORS: mix (our network and search together) would play %s." % label(f.mix_action)
        text += " The network alone: %s (%s)." % (label(f.net_action), _pct(q[f.net_action]))
        if f.shares is not None:
            legal = [a for a in f.actions if f.shares[a] > 0]
            if legal:
                best = max(legal, key=lambda a: (f.shares[a], -a))
                text += " The search alone: %s (%s of its visits)." % (label(best), _pct(f.shares[best]))
        else:
            text += " The search gave no answer this turn%s: only the network's numbers are there." % (
                " (%s)" % f.search_reason if f.search_reason else "")
        return text

    section("head", head)
    section("counts", counts)
    section("our_active", our_active)
    section("foe_active", foe_active)
    section("speed", speed)
    section("threats", lambda: _threats(b, q))
    section("our_bench", lambda: bench(True))
    section("foe_bench", lambda: bench(False))
    section("field", field_line)
    if getattr(f, "solo", False):
        section("foe_options", lambda: _foe_options(b, detail))
        section("earlier", lambda: _earlier(b, detail))
    if detail < 3:
        section("events", events)
    if getattr(f, "solo", False):
        section("memory", lambda: _memory(b, detail))
    if not b.forced:
        section("foe_next", lambda: _foe_next(b))
    section("lines", lambda: _look_ahead(b, detail))
    if not getattr(f, "solo", False):
        section("advisors", advisors)
    return "\n".join(lines)


def build_request(battle: Any, facts: Facts, model: str = DEFAULT_MODEL, budget: int = TOKEN_BUDGET) -> Built:
    """
    The request for one decision: the brief as "input", and one choice question whose choices are
    exactly the legal actions (value "a<our action id>-<name>", the description with its facts).
    Written at full detail and, only if that is over `budget` tokens, with parts left out
    (Built.detail says how many steps: 1 the shorter lists, 2 the bench's moves and Tera notes,
    3 last turn's events and the moves' effects).
    """
    began = time.perf_counter()
    board = _Board(battle, facts)
    q = facts.merged()
    order = [a for k in range(4) for a in (k, 10 + k)] + list(range(4, 10))
    legal = [a for a in order if a < len(facts.mask) and facts.mask[a] > 0.5 and a in facts.actions]
    built: Optional[Built] = None
    for detail in range(4):
        choices: List[Dict[str, str]] = []
        values: Dict[str, int] = {}
        for a in legal:
            target, kw = facts.actions[a]
            try:
                if hasattr(target, "current_hp"):
                    value, text = _switch_choice(board, a, target, q, detail)
                else:
                    value, text = _move_choice(board, a, target, bool((kw or {}).get("terastallize")), q, detail)
            except Exception:
                if "choice %d" % a not in board.problems:
                    board.problems.append("choice %d" % a)
                name = "switch" if hasattr(target, "current_hp") else "move"
                value, text = "a%d-%s" % (a, name), "%s %s. %s" % (name.capitalize(), _id(getattr(target, "species", None) or getattr(target, "id", "")),
                                                                    _advisors(board, a, q))
            while value in values:
                value += "x"
            values[value] = a
            choices.append({"value": value, "description": text.rstrip()})
        brief = _brief(board, q, detail)
        # The note about lines only when there are some: a request without them is word for word the Luna arm's.
        question = {"type": "choice", "name": QUESTION_NAME,
                    "instructions": SOLO_INSTRUCTIONS if facts.solo else INSTRUCTIONS + (LINES_NOTE if facts.lines else ""),
                    "choices": choices}
        body = {"model": model, "input": brief, "questions": [question]}
        built = Built(body=body, values=values, brief=brief, question=question, tokens=request_tokens(body), detail=detail,
                      problems=list(board.problems))
        if built.tokens <= budget + (LINES_TOKENS if facts.lines else 0) + (SOLO_TOKENS if facts.solo else 0):
            break
    assert built is not None
    built.ms = round((time.perf_counter() - began) * 1000.0, 2)
    return built


# =============================================================================================
# Luna at one decision: ask, read the answer, the rails, the counts
# =============================================================================================

@dataclass
class LunaSettings:
    """
    timeout      seconds for the API's answer (--luna-timeout). A decision also never waits past
                 its own deadline: the wait is the smaller of this and the time the deadline leaves.
    min_wait     with less time than this left before the deadline Luna is not asked ("no_time")
    guard        the blunder guard (--luna-guard on|off): see blunder_guard
    save_briefs  the decision's note keeps the brief and the question as they were sent (never the key)
    """
    timeout: float = 1.2
    min_wait: float = 0.25
    guard: bool = True
    guard_low: float = 0.01
    guard_high: float = 0.60
    save_briefs: bool = True
    budget: int = TOKEN_BUDGET


COUNT_KEYS = ("luna_decisions", "luna_asked", "luna_answered", "luna_played", "luna_overruled", "luna_agreed_with_mix",
              "luna_changed", "luna_tokens", "luna_latency_n", "luna_output_tokens")


def new_counts() -> Dict[str, Any]:
    """The counts of one game (or of a run): see result_fields."""
    out: Dict[str, Any] = {k: 0 for k in COUNT_KEYS}
    out["luna_latency_ms"] = 0.0
    out["luna_latency_max"] = 0.0
    out["luna_fallbacks"] = {}
    out["luna_cost"] = 0.0             # dollars, summed from each decision's own cost (the model's prices)
    out["luna_model"] = None
    return out


def add_counts(counts: Dict[str, Any], info: Dict[str, Any]) -> None:
    """One decision's "luna" record into a game's (or a run's) counts."""
    if not counts.get("luna_fallbacks") and "luna_fallbacks" not in counts:
        counts.update(new_counts())
    counts["luna_decisions"] += 1
    if info.get("model") and not counts.get("luna_model"):
        counts["luna_model"] = str(info["model"])
    if info.get("solo"):
        counts["luna_solo"] = True
    if info.get("asked"):
        counts["luna_asked"] += 1
        counts["luna_tokens"] += int(info.get("input_tokens") or 0)
        counts["luna_output_tokens"] = int(counts.get("luna_output_tokens") or 0) + int(info.get("output_tokens") or 0)
        cost = info.get("cost_usd")
        if cost is None:               # a record without one (Luna's, before the models): input tokens at Luna's price
            cost = int(info.get("input_tokens") or 0) * USD_PER_MILLION_TOKENS / 1e6
        counts["luna_cost"] = float(counts.get("luna_cost") or 0.0) + float(cost)
    if info.get("latency_ms") is not None and info.get("asked"):
        counts["luna_latency_n"] += 1
        counts["luna_latency_ms"] += float(info["latency_ms"])
        counts["luna_latency_max"] = max(float(counts.get("luna_latency_max") or 0.0), float(info["latency_ms"]))
    if info.get("choice") is not None:
        counts["luna_answered"] += 1
        counts["luna_agreed_with_mix"] += int(info.get("choice") == info.get("mix"))
        if info.get("overruled"):
            counts["luna_overruled"] += 1
        else:
            counts["luna_played"] += 1
            counts["luna_changed"] += int(info.get("choice") != info.get("mix"))
    if info.get("fell_back"):
        why = str(info.get("why") or "unknown")
        counts["luna_fallbacks"][why] = counts["luna_fallbacks"].get(why, 0) + 1


def result_fields(counts: Dict[str, Any]) -> Dict[str, Any]:
    """
    What a game's line in results.jsonl says about Luna: decisions with a choice (luna_decisions),
    questions sent (luna_asked), legal answers (luna_answered), answers played (luna_played),
    answers the blunder guard overruled (luna_overruled), answers that were mix's own pick
    (luna_agreed_with_mix), played answers that were not (luna_changed), the decisions mix played
    because of a rail, by reason (luna_fallbacks), the mean and slowest latency, the input tokens
    and what they cost.
    """
    n = int(counts.get("luna_latency_n") or 0)
    tokens = int(counts.get("luna_tokens") or 0)
    return {
        "luna_decisions": int(counts.get("luna_decisions") or 0), "luna_asked": int(counts.get("luna_asked") or 0),
        "luna_answered": int(counts.get("luna_answered") or 0), "luna_played": int(counts.get("luna_played") or 0),
        "luna_overruled": int(counts.get("luna_overruled") or 0),
        "luna_agreed_with_mix": int(counts.get("luna_agreed_with_mix") or 0), "luna_changed": int(counts.get("luna_changed") or 0),
        "luna_fallbacks": dict(counts.get("luna_fallbacks") or {}),
        "luna_latency_ms_mean": round(float(counts.get("luna_latency_ms") or 0.0) / n, 1) if n else None,
        "luna_latency_ms_max": round(float(counts.get("luna_latency_max") or 0.0), 1) if n else None,
        "luna_tokens": tokens,
        "luna_cost_usd": round(float(counts["luna_cost"]) if counts.get("luna_cost") else tokens * USD_PER_MILLION_TOKENS / 1e6, 6),
        "luna_output_tokens": int(counts.get("luna_output_tokens") or 0), "luna_model": counts.get("luna_model"),
        **({"luna_solo": True} if counts.get("luna_solo") else {}),
    }


def counts_text(counts: Dict[str, Any], label: str = "Luna") -> str:
    """One line: "Luna: asked 412 of 430 decisions with a choice, answered 398, ..." """
    r = result_fields(counts)
    fell = r["luna_fallbacks"]
    if r.get("luna_solo"):
        # The model alone: every answer is its own move, and what plays without one is the rule (solo_rule).
        text = ("%s alone: asked on %d of %d decisions with a choice, answered and played %d"
                % (label, r["luna_asked"], r["luna_decisions"], r["luna_played"]))
        if r["luna_answered"]:
            text += "; the fallback rule would have played the same on %d (%.0f%%)" % (
                r["luna_agreed_with_mix"], 100.0 * r["luna_agreed_with_mix"] / r["luna_answered"])
        text += "; the fallback rule played instead on %d" % sum(fell.values())
    else:
        text = ("%s: asked on %d of %d decisions with a choice, answered %d (played %d, overruled by the blunder guard %d)"
                % (label, r["luna_asked"], r["luna_decisions"], r["luna_answered"], r["luna_played"], r["luna_overruled"]))
        if r["luna_answered"]:
            text += "; its pick was mix's own on %d (%.0f%%), another action on %d" % (
                r["luna_agreed_with_mix"], 100.0 * r["luna_agreed_with_mix"] / r["luna_answered"],
                r["luna_answered"] - r["luna_agreed_with_mix"])
        text += "; mix played instead on %d" % sum(fell.values())
    if fell:
        text += " (" + ", ".join("%s %d" % kv for kv in sorted(fell.items())) + ")"
    if r["luna_latency_ms_mean"] is not None:
        text += "; answers took %.0f ms on average, %.0f ms at the slowest" % (r["luna_latency_ms_mean"], r["luna_latency_ms_max"])
    text += "; %s input tokens" % "{:,}".format(r["luna_tokens"])
    if r["luna_output_tokens"]:
        text += " and %s output tokens" % "{:,}".format(r["luna_output_tokens"])
    text += ", about $%.4f" % r["luna_cost_usd"]
    return text


class Luna:
    """
    The decider the search player holds for its "luna" battles: the client and the settings.
    build() writes the request of a decision; choose() asks it and applies the rails. The counts
    are the player's (search_player's game and run totals, through add_counts).
    """

    def __init__(self, client: LunaClient, settings: Optional[LunaSettings] = None, spec: Optional[ModelSpec] = None):
        self.client = client
        self.settings = settings or LunaSettings()
        self.spec = spec or getattr(client, "spec", None) or MODELS["luna"]
        self.brief_errors = 0

    @property
    def label(self) -> str:
        return self.spec.label

    def build(self, battle: Any, facts: Facts) -> Built:
        return build_request(battle, facts, model=self.client.model, budget=self.settings.budget)

    @staticmethod
    def info(facts: Facts, built: Optional[Built] = None) -> Dict[str, Any]:
        """A decision's "luna" record before anything was asked."""
        return {"asked": False, "mix": int(facts.mix_action), "choice": None, "confidence": None, "probabilities": None,
                "latency_ms": None, "input_tokens": built.tokens if built is not None else None, "tokens_from": "estimate",
                "fell_back": False, "why": None, "overruled": False}

    def stamp(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """A decision's record with whose it is: the model's name ("luna", "sol") and the API's model id."""
        info.setdefault("who", self.spec.name)
        info.setdefault("model", self.client.model)
        if info.get("solo"):
            info.pop("overruled_why", None)
        if info.get("asked") and info.get("cost_usd") is None:
            # Asked, and no usage came back (an error, a timeout): the input is charged all the same.
            info["cost_usd"] = round(cost_usd(self.spec, info.get("input_tokens")), 7)
        return info

    @staticmethod
    def fall_back(facts: Facts, why: str, built: Optional[Built] = None) -> Tuple[int, Dict[str, Any]]:
        """Mix's pick, for a reason that came before any question was sent."""
        info = Luna.info(facts, built)
        info.update(fell_back=True, why=str(why))
        return int(facts.mix_action), info

    async def choose(self, built: Built, facts: Facts, time_left: float) -> Tuple[int, Dict[str, Any]]:
        """
        (the action to play, the decision's "luna" record). time_left: seconds until this decision
        has to be an order. Never raises for anything the API does; mix's pick is the answer to
        every failure.
        """
        s = self.settings
        info = self.info(facts, built)
        if built.detail:
            info["detail"] = built.detail
        if built.problems:
            info["brief_problems"] = list(built.problems)
        mix_action = int(facts.mix_action)
        ok, why = self.client.available()
        if not ok:
            info.update(fell_back=True, why=why)
            return mix_action, info
        if float(time_left) < float(s.min_wait):
            info.update(fell_back=True, why="no_time", time_left_ms=round(1000.0 * max(0.0, float(time_left)), 1))
            return mix_action, info
        timeout = min(float(s.timeout), float(time_left))
        info.update(asked=True, timeout_ms=round(1000.0 * timeout, 1))
        try:
            reply = await self.client.ask(built.body, timeout)
        except LunaError as error:
            self.client.failed(error)
            info.update(fell_back=True, why=error.reason, error=error.text, attempts=error.attempts,
                        latency_ms=None if error.latency_ms is None else round(error.latency_ms, 1))
            if error.status is not None:
                info["http_status"] = error.status
            return mix_action, info
        info.update(latency_ms=reply.latency_ms, attempts=reply.attempts)
        if reply.request_id:
            info["request_id"] = _one_line(reply.request_id, 80)
        if reply.input_tokens is not None:
            info.update(input_tokens=int(reply.input_tokens), tokens_from="usage")
        if getattr(reply, "output_tokens", None) is not None:
            info["output_tokens"] = int(reply.output_tokens)
        if getattr(reply, "cached_tokens", None):
            info["cached_tokens"] = int(reply.cached_tokens)
        if getattr(reply, "reasoning_tokens", None):
            info["reasoning_tokens"] = int(reply.reasoning_tokens)
        info["cost_usd"] = round(cost_usd(self.spec, info.get("input_tokens"), info.get("output_tokens"), info.get("cached_tokens")), 7)
        if getattr(reply, "cost_usd", None) is not None:
            info["cost_usd"] = round(float(reply.cost_usd), 7)
        if getattr(reply, "served_by", None):
            info["served_by"] = _one_line(reply.served_by, 60)
        parsed = parse_answer(reply.answers, QUESTION_NAME, built.values)
        if parsed["kind"] != "choice":
            if parsed["kind"] == "malformed":
                self.client.failed(LunaError("malformed", parsed["why"], status=200))
            else:
                self.client.succeeded()              # the API works: it answered, with something we do not play
            info.update(fell_back=True, why=parsed["kind"], error=redact(parsed.get("why") or ""))
            if parsed.get("value") is not None:
                info["choice_value"] = redact(parsed["value"])
            return mix_action, info
        self.client.succeeded()
        action = int(parsed["action"])
        info.update(choice=action, choice_value=parsed["value"], confidence=parsed["confidence"],
                    probabilities={str(a): p for a, p in sorted(parsed["probabilities"].items())})
        if parsed.get("reason"):
            info["reason"] = parsed["reason"]
        if s.guard and action != mix_action and not facts.solo:      # (alone, there is no adviser to guard with)
            why = blunder_guard(action, mix_action, facts.merged(), facts.shares, s.guard_low, s.guard_high)
            if why:
                info.update(overruled=True, overruled_why=why)
                return mix_action, info
        return action, info


# =============================================================================================
# luna-check: one tiny real question before any game
# =============================================================================================

def check_body(model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """The check's request: a predicate ("is 2+2=4") and a 3-choice, about a one-line input."""
    return {
        "model": model,
        "input": "A check of the connection. The sum in question is 2 + 2 = 4.",
        "questions": [
            {"type": "predicate", "name": "two_plus_two", "instructions": "Is it true that 2 + 2 = 4?"},
            {"type": "choice", "name": "sum", "instructions": "What is 2 + 2?",
             "choices": [{"value": "three", "description": "The number 3."},
                         {"value": "four", "description": "The number 4."},
                         {"value": "five", "description": "The number 5."}]},
        ],
    }


SAMPLE_FILE = os.path.join(HERE, "luna_check_sample.json")
SAMPLE_NAME = "sample"


def sample_body(model: str = DEFAULT_MODEL, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    A question of a game's size and shape for the check: the request of one real logged decision
    (luna_check_sample.json, made by bench/luna_sample.py --json from the ladder games of the
    tests' fixture), with today's instructions and the question named "sample". None when the
    file is not there or is not a request.
    """
    try:
        with open(path or SAMPLE_FILE, encoding="utf-8") as fh:
            body = json.load(fh)
        question = dict(body["questions"][0])
        choices = [{"value": str(c["value"]), "description": str(c["description"])} for c in question["choices"]]
        if len(choices) < 2 or not isinstance(body.get("input"), str):
            return None
        return {"model": model, "input": body["input"],
                "questions": [{"type": "choice", "name": SAMPLE_NAME, "instructions": INSTRUCTIONS, "choices": choices}]}
    except Exception:
        return None


def read_check(answers: Any, probabilities: bool = True) -> Tuple[bool, List[str]]:
    """(the answers have the documented shape, what came back in plain lines). probabilities: whether a choice's answer documents a "probabilities" list (the Decisions API does; Sol's answer through the Responses API has none)."""
    lines: List[str] = []
    ok = True
    if not isinstance(answers, list):
        return False, ["\"answers\" is not a list"]
    by_name = {a.get("name"): a for a in answers if isinstance(a, dict)}
    pred = by_name.get("two_plus_two")
    if not isinstance(pred, dict):
        ok = False
        lines.append("no answer named two_plus_two")
    elif pred.get("type") == "refusal":
        ok = False
        lines.append("two_plus_two: the model refused")
    else:
        p = pred.get("probability")
        good = pred.get("type") == "predicate" and isinstance(p, (int, float)) and not isinstance(p, bool) and 0.0 <= float(p) <= 1.0
        ok = ok and good
        lines.append("predicate two_plus_two (is 2+2=4): %s" % ("probability %.2f" % float(p) if good else
                                                                "NOT the documented shape (type %s, probability %s)"
                                                                % (_one_line(pred.get("type"), 20), _one_line(p, 20))))
    choice = by_name.get("sum")
    if not isinstance(choice, dict):
        ok = False
        lines.append("no answer named sum")
    else:
        parsed = parse_answer(answers, "sum", {"three": 3, "four": 4, "five": 5})
        if parsed["kind"] == "choice":
            names = {3: "three", 4: "four", 5: "five"}
            probs = ", ".join("%s %.2f" % (names[a], p) for a, p in sorted(parsed["probabilities"].items()))
            lines.append("choice sum (what is 2+2): %s%s%s" % (
                parsed["value"], "" if parsed["confidence"] is None else ", confidence %.2f" % parsed["confidence"],
                (", probabilities " + probs) if probs else (", NO probabilities" if probabilities else "")))
            if not parsed["probabilities"] and probabilities:
                lines.append("note: the answer has no \"probabilities\" list (the documented answer has one); the choice itself is usable")
        else:
            ok = False
            lines.append("choice sum: %s (%s)" % (parsed["kind"], _one_line(parsed.get("why"), 80)))
    return ok, lines


def luna_check(client: LunaClient, timeout: float = 15.0, game_timeout: float = 1.2, out: Callable[[str], None] = print,
               repeats: int = 2, sample: Any = True) -> int:
    """
    The check: the key, the endpoint and the answer's shape, with one tiny real request (is
    2+2=4, and a 3-choice). Then, on the connection now open, a question of a game's size and
    shape (sample_body: a real logged decision, about 13 choices and 1,900 tokens), `repeats`
    times: that the API takes a request that large, and how long its answer takes, which is what
    a game will see. Prints what came back, never the key. 0 when Luna can be played, 1 when not
    (the reason in one line). sample: True for the file's, a request body to use instead, or
    None to leave that part out.
    """
    body = check_body(client.model)
    name = str(getattr(client, "label", "") or "Luna")
    failed = name.upper() + " CHECK FAILED"
    flag = getattr(client, "timeout_flag", None) or ("--luna-timeout" if name == "Luna" else "--%s-timeout" % name.lower())
    out("%s check: one tiny question to %s (model %s), key ***" % (name, client.url, client.model))
    try:
        first = client.ask_sync(body, timeout)
    except LunaError as error:
        out("%s: %s" % (failed, error.text))
        return 1
    ok, lines = read_check(first.answers, probabilities=getattr(client, "api", "decisions") == "decisions")
    out("  HTTP 200 in %.0f ms (a new connection)" % first.latency_ms)
    for line in lines:
        out("  " + line)
    if not ok:
        out("%s: the answers are not what the API documents (see above): %s cannot be played with them" % (failed, name))
        return 1
    big = sample_body(client.model) if sample is True else sample
    warm: List[float] = []
    if big is not None:
        question = big["questions"][0]
        values = {c["value"]: i for i, c in enumerate(question["choices"])}
        tokens = request_tokens(big)
        for i in range(max(1, int(repeats))):
            try:
                reply = client.ask_sync(big, timeout)
            except LunaError as error:
                out("%s: a question of a game's size (%d choices, about %d tokens) was not answered: %s"
                    % (failed, len(values), tokens, error.text))
                return 1
            parsed = parse_answer(reply.answers, question["name"], values)
            if parsed["kind"] != "choice":
                out("%s: a question of a game's size (%d choices, about %d tokens) got no usable answer: %s (%s)"
                    % (failed, len(values), tokens, parsed["kind"], _one_line(parsed.get("why"), 80)))
                return 1
            warm.append(reply.latency_ms)
            if i == 0:
                out("  a question of a game's size (%d choices, about %d tokens): HTTP 200, choice %s%s"
                    % (len(values), tokens, parsed["value"],
                       "" if parsed["confidence"] is None else ", confidence %.2f" % parsed["confidence"]))
                if parsed.get("reason"):
                    out("  its reason: %s" % parsed["reason"])
                if getattr(reply, "output_tokens", None) is not None:
                    spec = getattr(client, "spec", None)
                    out("  tokens: %s in, %s out%s: about $%.4f a decision" % (
                        reply.input_tokens, reply.output_tokens,
                        (" (%d of them reasoning)" % reply.reasoning_tokens) if getattr(reply, "reasoning_tokens", None) else "",
                        cost_usd(spec, reply.input_tokens, reply.output_tokens, reply.cached_tokens)))
    else:
        for _ in range(max(0, int(repeats))):
            try:
                warm.append(client.ask_sync(body, timeout).latency_ms)
            except LunaError as error:
                out("  a second question failed: %s" % error.text)
                break
    if warm:
        fastest = min(warm)
        out("  its answers on the open connection took: %s ms" % ", ".join("%.0f" % ms for ms in warm))
        if fastest > 1000.0 * game_timeout:
            out("  WARNING: slower than the %.1f s a game gives it (%s): most decisions would fall back to today's bot" % (game_timeout, flag))
        elif fastest > 600.0 * game_timeout:
            out("  note: more than half of the %.1f s a game gives it: some decisions will fall back to today's bot" % game_timeout)
    if getattr(client, "effort_fell_back", None):
        out("  note: the API refused reasoning effort \"%s\"; \"%s\" is used" % (client.effort_fell_back, client.effort))
    out("%s check OK: the key works, the endpoint answers, and the answers have the documented shape" % name)
    return 0
