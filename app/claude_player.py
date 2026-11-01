"""
Claude (Anthropic) as a model that plays alone: the arms "fable-solo", "opus-solo", "sonnet-solo"
and "haiku-solo" of search_player.SOLO_ARMS.

    python ladder.py --player fable-solo --games 5        Claude Fable 5.1 picks every move
    python ladder.py --luna-check --model opus            one tiny real question to Claude Opus 5.5

The question is the one luna_player builds for a model alone (the position as text, the legal
actions as choices): nothing of it is written here. This module is the wire: the same question as
one request to the Messages API through Anthropic's own SDK, and its answer read back in the shape
luna_player.Luna already plays (LunaReply). ClaudeClient has LunaClient's interface, so the
decider, the rails, the counts and the spending cap are the ones of the other models.

How the request is made (the Messages API as these models take it):
  * No "thinking" field. Fable 5.1 and Opus 5.5 always think and reject any other setting, and
    Sonnet 5 thinks unless told not to. How much is output_config.effort ("low" here, so that a
    move usually stays under TARGET_SECONDS; the wait itself ends only at DECISION_SECONDS).
    Haiku 4.5 gets no effort and does not think.
  * The answer is JSON by output_config.format (a schema), not a forced tool call: Fable 5.1 and
    Opus 5.5 answer a forced tool_choice with HTTP 400.
  * A game's question always has the same schema (the choice is one of a0 to a13, whichever of
    them are legal): a schema is compiled on first use, and a new one per move would cost that
    every time.
  * The instructions are the system prompt, marked for the prompt cache (they are the same at
    every move; a prefix under the model's minimum is simply not cached).
  * A refusal (stop_reason "refusal") is an answer of type "refusal": the fallback rule plays.
    Fable 5.1 is asked with server-side fallbacks (beta server-side-fallback-2026-06-01, to
    claude-opus-5), so a declined request is answered by that model in the same call. If the API
    rejects that form the client goes on without it, said once.

The key (ANTHROPIC_API_KEY, or the line the window app writes) is held in memory, given to the
SDK and to nothing else, and never printed: every error text passes luna_player.redact.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import getpass
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import luna_player as lp
from luna_player import LunaError, LunaReply, ModelSpec

__all__ = ["MODELS", "SOLO_ARMS", "KEY_ENV", "BASE_URL", "API", "ClaudeClient", "read_key", "endpoint_problem",
           "messages_request", "messages_answers", "request_cost_usd", "DECISION_SECONDS", "TARGET_SECONDS", "EFFORTS"]

API = "anthropic"
KEY_ENV = "ANTHROPIC_API_KEY"
BASE_URL = "https://api.anthropic.com"
TARGET_SECONDS = 10.0               # what a move should stay under: the reason for effort "low" (not a cut-off)
DECISION_SECONDS = 25.0             # the most one move waits for the model; after it the fallback rule plays
EFFORTS = ("low", "medium", "high")
MAX_TOKENS = 2048                   # the thinking and the answer together (the answer is some 80 tokens)
FALLBACK_BETA = "server-side-fallback-2026-06-01"
CACHE_WRITE_FACTOR = 1.25           # a prompt-cache write costs this many times the input price

# name, label, the API's model id, api, address, $ per 1M tokens (input, cache read, output), the
# seconds a game gives one answer, output_config.effort ("" = the field is not sent).
MODELS: Dict[str, ModelSpec] = {
    "fable": ModelSpec("fable", "Claude Fable 5.1", "claude-fable-5-1", API, BASE_URL, 10.0, 0.25, 50.0, DECISION_SECONDS, effort="low"),
    "opus": ModelSpec("opus", "Claude Opus 5.5", "claude-opus-5-5", API, BASE_URL, 4.0, 0.20, 20.0, DECISION_SECONDS, effort="low"),
    "sonnet": ModelSpec("sonnet", "Claude Sonnet 5", "claude-sonnet-5", API, BASE_URL, 2.0, 0.20, 10.0, DECISION_SECONDS, effort="low"),
    "haiku": ModelSpec("haiku", "Claude Haiku 4.5", "claude-haiku-4-5", API, BASE_URL, 1.0, 0.10, 5.0, DECISION_SECONDS, effort=""),
}
SOLO_ARMS: Tuple[str, ...] = tuple(name + "-solo" for name in MODELS)
# The model a declined request is run on instead, inside the same call (Fable 5.1 only: the
# targets the API allows for the others are not settled).
REFUSAL_FALLBACK: Dict[str, str] = {"claude-fable-5-1": "claude-opus-5"}

FORMAT_NOTE = (
    "Answer with one JSON object that matches the schema, and nothing else. For a choice: first \"reason\", the deciding "
    "fact in at most 25 words; then \"choice\", exactly one of the offered ids (the text before the colon); then "
    "\"confidence\", from 0 to 1, that it is the best choice. For a yes/no question: \"probability\", from 0 to 1, that "
    "the answer is yes. The move is due within seconds, so decide quickly."
)
_ACTION_VALUE = re.compile(r"^a(\d+)-")
ACTION_IDS: List[str] = ["a%d" % i for i in range(lp.ACTIONS)]


def read_key(environ: Optional[Dict[str, str]] = None, ask: Optional[Callable[[str], str]] = None) -> str:
    """
    The Anthropic API key: ANTHROPIC_API_KEY when it is set, else typed once at a hidden prompt.
    Remembered for luna_player.redact. LunaError("key") when there is none or it cannot be one.
    """
    environ = os.environ if environ is None else environ
    key = str(environ.get(KEY_ENV) or "").strip().strip("\"'")
    if not key:
        try:
            key = str((ask or getpass.getpass)("Anthropic API key (not shown, not stored): ") or "").strip().strip("\"'")
        except EOFError:
            key = ""
    if not key:
        raise LunaError("key", "no Anthropic API key: set %s or type the key at the prompt" % KEY_ENV)
    if any(ord(c) < 33 or ord(c) > 126 for c in key):
        lp.remember_secret(key)
        raise LunaError("key", "the key has a space, a line break or another character a key cannot have: paste it again")
    lp.remember_secret(key)
    return key


def endpoint_problem(url: str) -> str:
    """Why the key must not be sent to this address, or "": Anthropic's API (https) and a test server on this machine only."""
    from urllib.parse import urlparse

    try:
        parts = urlparse(str(url))
    except Exception:
        return "not an address"
    host = (parts.hostname or "").lower()
    if parts.scheme == "https" and host == "api.anthropic.com":
        return ""
    if parts.scheme in ("http", "https") and host in ("127.0.0.1", "localhost", "::1"):
        return ""
    return "the key is sent to Anthropic (https://api.anthropic.com) or to a test server on this machine only, not to %s" % (host or "?")


def messages_request(body: Dict[str, Any], spec: ModelSpec, effort: str = "") -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """
    A Decisions-shaped request (the input, the questions) as the arguments of messages.create that
    ask the same, and for each choice question the map from the id the model answers with to the
    value we offered. A question whose values are all "a<action id>-..." (a game's) is offered by
    its ids a0 to a13 and always has the same schema; any other by its own values.
    """
    questions = [q for q in body.get("questions") or [] if isinstance(q, dict)]
    single = len(questions) == 1
    system = FORMAT_NOTE
    source = body.get("input")
    parts: List[str] = [source if isinstance(source, str) else json.dumps(source, ensure_ascii=False)]
    properties: Dict[str, Any] = {}
    required: List[str] = []
    maps: Dict[str, Dict[str, str]] = {}
    for q in questions:
        name = str(q.get("name") or "answer")
        instructions = str(q.get("instructions") or "")
        if single and instructions:
            system = instructions + "\n\n" + FORMAT_NOTE
        if q.get("type") == "choice":
            choices = [c for c in q.get("choices") or [] if isinstance(c, dict)]
            values = [str(c.get("value")) for c in choices]
            short: Dict[str, str] = {}
            for value in values:
                m = _ACTION_VALUE.match(value)
                if m is None or int(m.group(1)) >= lp.ACTIONS or ("a" + m.group(1)) in short:
                    short = {}
                    break
                short["a" + m.group(1)] = value
            game = bool(short) and len(short) == len(values)
            maps[name] = short if game else {value: value for value in values}
            ids = {value: key for key, value in maps[name].items()}
            head = "QUESTION \"%s\"%s\nCHOICES:" % (name, "" if single or not instructions else ": " + instructions)
            parts.append(head + "\n" + "\n".join("- %s: %s" % (ids[str(c.get("value"))], c.get("description")) for c in choices))
            properties[name] = {"type": "object", "additionalProperties": False, "required": ["reason", "choice", "confidence"],
                                "properties": {"reason": {"type": "string"},
                                               "choice": {"type": "string", "enum": list(ACTION_IDS) if game else values},
                                               "confidence": {"type": "number"}}}
        else:
            parts.append("QUESTION \"%s\" (yes or no)%s" % (name, "" if single or not instructions else ": " + instructions))
            properties[name] = {"type": "object", "additionalProperties": False, "required": ["probability"],
                                "properties": {"probability": {"type": "number"}}}
        required.append(name)
    output_config: Dict[str, Any] = {"format": {"type": "json_schema", "schema": {
        "type": "object", "additionalProperties": False, "required": required, "properties": properties}}}
    if effort:
        output_config["effort"] = str(effort)
    request: Dict[str, Any] = {
        "model": spec.model,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "\n\n".join(parts)}],
        "output_config": output_config,
    }
    return request, maps


def messages_answers(message: Any, body: Dict[str, Any], maps: Dict[str, Dict[str, str]]) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    (the "answers" of a Messages reply in the Decisions shape, "") or (None, why it has none). A
    refusal is an answer of type "refusal" for every question. A choice that is none of the ids
    offered is passed on as it came, for the caller to call illegal.
    """
    questions = [q for q in body.get("questions") or [] if isinstance(q, dict)]
    stop = getattr(message, "stop_reason", None)
    if stop == "refusal":
        return [{"type": "refusal", "name": q.get("name")} for q in questions], ""
    texts = [str(block.text) for block in getattr(message, "content", None) or []
             if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)]
    if not texts:
        return None, "no text in the reply (stop reason %s)" % lp._one_line(stop, 20)
    try:
        obj = json.loads("".join(texts))
    except ValueError:
        return None, "the reply's text is not JSON (%d characters, stop reason %s)" % (len("".join(texts)), lp._one_line(stop, 20))
    if not isinstance(obj, dict):
        return None, "the reply is not a JSON object"
    answers: List[Dict[str, Any]] = []
    for q in questions:
        name = q.get("name")
        got = obj.get(name)
        if not isinstance(got, dict):
            return None, "no answer named %s in the reply" % lp._one_line(name, 30)
        if q.get("type") == "choice":
            choice = got.get("choice")
            if isinstance(choice, str):
                choice = maps.get(str(name), {}).get(choice, choice)
            answers.append({"type": "choice", "name": name, "choice": choice, "confidence": got.get("confidence"),
                            "probabilities": [], "reason": got.get("reason")})
        else:
            answers.append({"type": "predicate", "name": name, "probability": got.get("probability")})
    return answers, ""


def _count(value: Any) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def request_cost_usd(spec: ModelSpec, usage: Any) -> Tuple[float, Dict[str, int]]:
    """
    What one answered request cost at the model's prices, and its token counts: "input" is every
    input token (fresh, read from the cache, written to it), "cached" the ones read from the cache,
    "output" the answer's and the thinking's. A cache write is charged at 1.25 times the input price.
    """
    fresh = _count(getattr(usage, "input_tokens", None))
    read = _count(getattr(usage, "cache_read_input_tokens", None))
    written = _count(getattr(usage, "cache_creation_input_tokens", None))
    out = _count(getattr(usage, "output_tokens", None))
    usd = (fresh * spec.usd_in + read * spec.usd_cached + written * spec.usd_in * CACHE_WRITE_FACTOR + out * spec.usd_out) / 1e6
    return usd, {"input": fresh + read + written, "cached": read, "output": out}


class ClaudeClient:
    """
    The Messages API through the anthropic SDK, with LunaClient's interface.

    ask(body, timeout)       from the event loop: the request runs in a thread of this client's
                             own pool and is given up after `timeout` seconds.
    ask_sync(body, timeout)  the request itself. One retry of a failed connection or an HTTP 5xx
                             when at least retry_min_left s are left. Never of a timeout, a 4xx
                             or a 429. The SDK's own retries are off: a move cannot wait for them.
    available()              (False, "paused") for breaker_pause s after breaker_failures failures
                             in a row; (False, "key_refused") for the rest of the run once the API
                             refused the key (401, 403) or the model (404).
    succeeded() / failed(e)  the caller says how each request went.
    """

    RETRY_STATUS = (500, 502, 503, 504, 529)
    FATAL_STATUS = (401, 403, 404)

    def __init__(self, key: str, spec: ModelSpec, effort: Optional[str] = None, retries: int = 1, retry_min_left: float = 1.0,
                 breaker_failures: int = 5, breaker_pause: float = 60.0, clock: Callable[[], float] = time.monotonic,
                 say: Optional[Callable[[str], None]] = None, workers: int = 8, refusal_fallback: bool = True):
        why = endpoint_problem(spec.url)
        if why:
            raise LunaError("endpoint", why)
        if not key:
            raise LunaError("key", "no Anthropic API key")
        lp.remember_secret(key)
        import anthropic

        self._sdk = anthropic
        # The SDK's own retries are off (max_retries=0): the caller's deadline decides, and one
        # retry is made here when there is time for it.
        self._client = anthropic.Anthropic(api_key=key, base_url=str(spec.url), max_retries=0, timeout=float(spec.timeout) + 5.0)
        self.spec = spec
        self.url = str(spec.url)
        self.model = str(spec.model)
        self.api = API
        self.label = str(spec.label)
        self.effort = str(spec.effort if effort is None else effort)
        self.effort_fell_back: Optional[str] = None       # the effort the API refused, once it did
        self.fallback_model: Optional[str] = REFUSAL_FALLBACK.get(self.model) if refusal_fallback else None
        self.fallback_refused: Optional[str] = None        # why the API did not take the fallbacks form, once it did not
        self.retries = max(0, int(retries))
        self.retry_min_left = float(retry_min_left)
        self.breaker_failures = max(1, int(breaker_failures))
        self.breaker_pause = float(breaker_pause)
        self._clock = clock
        self.say = say or (lambda text: print(text, flush=True))
        self._failures = 0
        self._paused_until: Optional[float] = None
        self.pauses = 0
        self.disabled: Optional[str] = None
        self.off_reason = "key_refused"
        self.last_error: Optional[str] = None
        self.requests_sent = 0
        self.timeout_flag = "--claude-timeout"
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="claude")

    def __repr__(self) -> str:
        return "ClaudeClient(url=%r, model=%r, key=***)" % (self.url, self.model)

    # -- the request --------------------------------------------------------------------------------

    def _create(self, request: Dict[str, Any], left: float) -> Any:
        client = self._client.with_options(timeout=max(0.05, float(left)), max_retries=0)
        if self.fallback_model:
            try:
                return client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks=[{"model": self.fallback_model}], **request)
            except TypeError as exc:           # an SDK from before the parameter: asked without it
                self.fallback_refused = "this version of the anthropic SDK has no fallbacks parameter (%s)" % lp._one_line(exc, 80)
                self.fallback_model = None
                self.say("%s: %s: asked without the refusal fallback." % (self.label, self.fallback_refused))
        return client.messages.create(**request)

    def ask_sync(self, body: Dict[str, Any], timeout: float) -> LunaReply:
        """One question, answered within `timeout` seconds or LunaError. Blocking: call it from a thread."""
        sdk = self._sdk
        began = time.monotonic()
        deadline = began + float(timeout)
        request, maps = messages_request(body, self.spec, self.effort)
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
                message = self._create(request, left)
            except sdk.APITimeoutError:
                raise LunaError("timeout", "no answer within %.0f ms" % (1000.0 * timeout), attempts=attempts,
                                latency_ms=(time.monotonic() - began) * 1000.0)
            except sdk.APIStatusError as exc:
                status = int(getattr(exc, "status_code", 0) or 0)
                text = lp._one_line(getattr(exc, "message", None) or exc, 220)
                error = LunaError("http", "HTTP %d: %s" % (status, text), status=status, attempts=attempts,
                                  latency_ms=(time.monotonic() - began) * 1000.0)
                again = status in self.RETRY_STATUS
                changed = False
                if status == 400 and self.fallback_model and re.search(r"fallback|beta", text, re.I):
                    # The API does not take the server-side fallbacks form (for this model or this
                    # account): asked without it from here on. A declined request is then the rule's.
                    self.fallback_refused = lp.redact(text)
                    self.fallback_model = None
                    self.say("%s: the API did not take the refusal fallback (%s): asked without it from here on."
                             % (self.label, self.fallback_refused))
                    changed = True
                elif status == 400 and self.effort and re.search(r"effort", text, re.I):
                    self.effort_fell_back = self.effort
                    self.effort = "" if self.effort == "low" else "low"
                    self.say("%s: the API refused effort \"%s\" (%s): %s from here on."
                             % (self.label, self.effort_fell_back, lp.redact(text),
                                "\"low\" is used" if self.effort else "no effort is sent"))
                    request, maps = messages_request(body, self.spec, self.effort)
                    changed = True
                if changed:
                    attempts -= 1
                    if deadline - time.monotonic() >= self.retry_min_left:
                        continue
            except sdk.APIConnectionError as exc:
                error = LunaError("network", "%s: %s" % (type(exc).__name__, lp._one_line(exc, 200)), attempts=attempts,
                                  latency_ms=(time.monotonic() - began) * 1000.0)
                again = True
            else:
                latency = (time.monotonic() - began) * 1000.0
                stop = getattr(message, "stop_reason", None)
                answers, why = messages_answers(message, body, maps)
                if answers is None:
                    if stop == "max_tokens":
                        why = "the answer was cut off at %d tokens (the thinking and the answer together)" % MAX_TOKENS
                    raise LunaError("malformed", "HTTP 200 without a usable answer: %s" % why, status=200, attempts=attempts, latency_ms=latency)
                usd, used = request_cost_usd(self.spec, getattr(message, "usage", None))
                reply = LunaReply(answers=answers, latency_ms=round(latency, 1), attempts=attempts, input_tokens=used["input"] or None,
                                  status=200, request_id=getattr(message, "_request_id", None), body=None,
                                  output_tokens=used["output"] or None, cached_tokens=used["cached"] or None)
                reply.cost_usd = usd
                served = str(getattr(message, "model", "") or "")
                if self.fallback_model and served.startswith(self.fallback_model):
                    reply.served_by = served          # the first model declined: the fallback's answer
                return reply
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
        """An API failure. A timeout is not counted towards the pause: a slow answer is not a broken API."""
        self.last_error = error.text
        if error.kind == "http" and error.status in self.FATAL_STATUS:
            if not self.disabled:
                self.disabled = error.text
                self.say("%s is OFF for the rest of this run: the API refused the key or the model (%s). The fallback rule "
                         "plays the games in progress, and no new games are started." % (self.label, error.text))
            return
        if error.kind == "timeout":
            return
        self._failures += 1
        if self._failures >= self.breaker_failures:
            self._failures = 0
            self.pauses += 1
            self._paused_until = self._clock() + self.breaker_pause
            self.say("%s: %d API failures in a row (the last: %s). Not asked for %.0f s: the fallback rule plays meanwhile."
                     % (self.label, self.breaker_failures, error.text, self.breaker_pause))

    def close(self) -> None:
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass
