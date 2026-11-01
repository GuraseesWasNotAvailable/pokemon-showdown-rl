"""
A fake of Anthropic's Messages API for the tests of claude_player (POST /v1/messages on 127.0.0.1).

It answers the question claude_player.messages_request writes, and it REFUSES what the real API
refuses for these models, so that a request of the wrong shape fails a test and not a game:

  * x-api-key must be the fake's key (else 401), anthropic-version must be there (else 400)
  * the model must be one of MODEL_IDS (else 404)
  * max_tokens and a first message of role "user" are required
  * "thinking": Fable 5.1 and Opus 5.5 take only {"type": "adaptive"} or none (else 400)
  * tool_choice "any" / "tool": 400 on Fable 5.1 and Opus 5.5
  * temperature, top_p, top_k: 400 on Fable 5.1, Opus 5.5 and Sonnet 5
  * a last message of role "assistant" (prefill): 400
  * output_config.format must be a json_schema whose objects all say additionalProperties false
  * output_config.effort: one of low, medium, high, xhigh, max; 400 on Haiku 4.5
  * "fallbacks" needs the path /v1/messages?beta=true and the beta header
    server-side-fallback-2026-06-01 (else 400); refuse_fallbacks=True answers it with 400 always

mode (or the script, one entry per request): "first" / "last" (that legal choice), "illegal" (an
id of the schema that was not offered), "refusal", "cut" (stop_reason max_tokens, no text),
"not_json", "hang" (hang_s seconds, then the answer), "http_401" ... "http_529", "served_by_fallback"
(the fallback model's answer). .requests keeps what came in (never the key itself).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Sequence

MODEL_IDS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
ALWAYS_THINKING = ("claude-fable-5-1", "claude-opus-5-5")
NO_SAMPLING = ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5")
NO_EFFORT = ("claude-haiku-4-5",)
FALLBACK_BETA = "server-side-fallback-2026-06-01"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _strict(schema: Any, where: str, problems: List[str]) -> None:
    if not isinstance(schema, dict):
        problems.append("%s is not an object" % where)
        return
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            problems.append("%s: additionalProperties must be false" % where)
        for name, sub in (schema.get("properties") or {}).items():
            _strict(sub, "%s.%s" % (where, name), problems)
        if sorted(schema.get("required") or []) != sorted((schema.get("properties") or {}).keys()):
            problems.append("%s: every property must be required" % where)


def check_request(body: Any, path: str, beta: str) -> List[str]:
    """Why the real API would answer this request with HTTP 400 ([] when it would not)."""
    problems: List[str] = []
    if not isinstance(body, dict):
        return ["the body is not a JSON object"]
    model = body.get("model")
    if not isinstance(body.get("max_tokens"), int) or body["max_tokens"] <= 0:
        problems.append("max_tokens: a positive integer is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict) or messages[0].get("role") != "user":
        problems.append("messages: the first message must have role user")
    elif messages[-1].get("role") == "assistant":
        problems.append("messages: a last message of role assistant (prefill) is not supported")
    thinking = body.get("thinking")
    if thinking is not None and model in ALWAYS_THINKING and (not isinstance(thinking, dict) or thinking.get("type") != "adaptive"):
        problems.append("\"thinking.type.%s\" is not supported for this model" % (thinking.get("type") if isinstance(thinking, dict) else thinking))
    if isinstance(thinking, dict) and "budget_tokens" in thinking and model != "claude-haiku-4-5":
        problems.append("thinking.budget_tokens is not supported for this model")
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") in ("any", "tool") and model in ALWAYS_THINKING:
        problems.append("tool_choice: type \"tool\" and \"any\" are not supported for this model.")
    for name in ("temperature", "top_p", "top_k"):
        if name in body and model in NO_SAMPLING:
            problems.append("%s is not supported for this model" % name)
    system = body.get("system")
    if system is not None and not isinstance(system, str):
        if not isinstance(system, list) or any(not isinstance(b, dict) or b.get("type") != "text" or not isinstance(b.get("text"), str)
                                               for b in system):
            problems.append("system: a string or a list of text blocks")
    config = body.get("output_config")
    if config is not None:
        if not isinstance(config, dict):
            problems.append("output_config is not an object")
        else:
            unknown = sorted(set(config) - {"format", "effort"})
            if unknown:
                problems.append("output_config: unknown field %s" % unknown[0])
            if "effort" in config:
                if config["effort"] not in EFFORTS:
                    problems.append("output_config.effort: not one of %s" % ", ".join(EFFORTS))
                elif model in NO_EFFORT:
                    problems.append("output_config.effort is not supported for this model")
            fmt = config.get("format")
            if fmt is not None:
                if not isinstance(fmt, dict) or fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
                    problems.append("output_config.format: {type: json_schema, schema: {...}}")
                else:
                    _strict(fmt["schema"], "schema", problems)
    if "fallbacks" in body:
        if "beta=true" not in path or FALLBACK_BETA not in [b.strip() for b in (beta or "").split(",")]:
            problems.append("fallbacks: this parameter needs the beta header %s" % FALLBACK_BETA)
        elif (not isinstance(body["fallbacks"], list) or not body["fallbacks"]
              or any(not isinstance(e, dict) or set(e) - {"model", "max_tokens"} or e.get("model") not in MODEL_IDS for e in body["fallbacks"])):
            problems.append("fallbacks: a list of {model} entries")
    return problems


def questions_of(body: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{question name: {"type", "enum" (the schema's), "offered" (the ids listed in the user's text)}}."""
    schema = (((body.get("output_config") or {}).get("format") or {}).get("schema") or {})
    text = ""
    for message in body.get("messages") or []:
        content = message.get("content")
        text += content if isinstance(content, str) else "".join(b.get("text", "") for b in content or [] if isinstance(b, dict))
    out: Dict[str, Dict[str, Any]] = {}
    for name, sub in (schema.get("properties") or {}).items():
        props = sub.get("properties") or {}
        if "choice" in props:
            m = re.search(r'QUESTION "%s"[^\n]*\nCHOICES:\n((?:- [^\n]*\n?)+)' % re.escape(name), text)
            offered = re.findall(r"^- ([^:\n]+):", m.group(1), re.M) if m else []
            out[name] = {"type": "choice", "enum": list(props["choice"].get("enum") or []), "offered": offered}
        else:
            out[name] = {"type": "predicate"}
    return out


class FakeAnthropic:
    def __init__(self, key: str = "sk-ant-fake-0000", mode: str = "first", hang_s: float = 3.0, refuse_fallbacks: bool = False,
                 cache_read: int = 0, cache_write: int = 0, thinking_tokens: int = 120, port: int = 0):
        self.key = key
        self.mode = mode
        self.hang_s = float(hang_s)
        self.refuse_fallbacks = bool(refuse_fallbacks)
        self.cache_read, self.cache_write, self.thinking_tokens = int(cache_read), int(cache_write), int(thinking_tokens)
        self.script: Deque[str] = deque()
        self.requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            wbufsize = 1 << 16

            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                fake._handle(self)

            def _send(self, status: int, payload: Any, raw: Optional[bytes] = None) -> None:
                data = raw if raw is not None else json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("request-id", "req_fake_%06d" % len(fake.requests))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                except Exception:
                    pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self.url = "http://127.0.0.1:%d" % self.port
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "FakeAnthropic":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass

    def __enter__(self) -> "FakeAnthropic":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @staticmethod
    def _error(status: int, message: str) -> Dict[str, Any]:
        kinds = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error",
                 429: "rate_limit_error", 500: "api_error", 529: "overloaded_error"}
        return {"type": "error", "error": {"type": kinds.get(status, "api_error"), "message": message}}

    def _handle(self, h: Any) -> None:
        began = time.time()
        length = int(h.headers.get("Content-Length") or 0)
        raw = h.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = None
        given = h.headers.get("x-api-key")
        beta = h.headers.get("anthropic-beta") or ""
        record: Dict[str, Any] = {"path": h.path, "key": "ok" if given == self.key else ("none" if not given else "wrong"),
                                  "version": h.headers.get("anthropic-version"), "beta": beta, "body": body, "problems": []}
        status, payload, raw_out = 200, None, None
        with self._lock:
            behaviour = self.script.popleft() if self.script else self.mode
        if not h.path.startswith("/v1/messages"):
            status, payload = 404, self._error(404, "Not found: %s" % h.path)
        elif record["key"] != "ok":
            status, payload = 401, self._error(401, "invalid x-api-key")
        elif not record["version"]:
            status, payload = 400, self._error(400, "anthropic-version: header is required")
        else:
            record["problems"] = check_request(body, h.path, beta)
            if record["problems"]:
                status, payload = 400, self._error(400, "; ".join(record["problems"][:3]))
            elif body.get("model") not in MODEL_IDS:
                status, payload = 404, self._error(404, "model: %s" % body.get("model"))
            elif "fallbacks" in body and self.refuse_fallbacks:
                status, payload = 400, self._error(400, "fallbacks: Extra inputs are not permitted for this model")
                behaviour = "fallbacks_refused"
        if status == 200:
            if behaviour == "hang":
                time.sleep(self.hang_s)
                behaviour = self.mode if self.mode != "hang" else "first"
                record["hung"] = True
            if behaviour.startswith("http_"):
                status = int(behaviour.split("_", 1)[1])
                payload = self._error(status, "the fake's HTTP %d" % status)
            else:
                model = body["model"]
                served = model
                if behaviour == "served_by_fallback" and body.get("fallbacks"):
                    served = body["fallbacks"][0]["model"]
                    behaviour = "first"
                thinks = model != "claude-haiku-4-5"
                content: List[Dict[str, Any]] = []
                if thinks:
                    content.append({"type": "thinking", "thinking": "", "signature": "sig_fake"})
                stop = "end_turn"
                text = ""
                if behaviour == "refusal":
                    stop, content = "refusal", []
                elif behaviour == "cut":
                    stop = "max_tokens"
                elif behaviour == "not_json":
                    text = "I would play Earthquake here."
                else:
                    obj: Dict[str, Any] = {}
                    for name, q in questions_of(body).items():
                        if q["type"] == "choice":
                            offered = q["offered"] or q["enum"]
                            pick = offered[-1] if behaviour == "last" else offered[0]
                            if behaviour == "illegal":
                                pick = next((e for e in q["enum"] if e not in offered), "zz")
                            obj[name] = {"reason": "The fake's reason for %s." % pick, "choice": pick, "confidence": 0.8}
                        else:
                            obj[name] = {"probability": 0.99}
                    text = json.dumps(obj)
                if text:
                    content.append({"type": "text", "text": text})
                tokens_in = max(1, len(raw) // 4)
                usage = {"input_tokens": max(1, tokens_in - self.cache_read - self.cache_write),
                         "cache_read_input_tokens": self.cache_read, "cache_creation_input_tokens": self.cache_write,
                         "output_tokens": (self.thinking_tokens if thinks else 0) + max(1, len(text) // 4)}
                payload = {"id": "msg_fake_%06d" % len(self.requests), "type": "message", "role": "assistant", "model": served,
                           "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage}
                if stop == "refusal":
                    payload["stop_details"] = {"type": "refusal", "category": None, "explanation": None}
        record.update(behaviour=behaviour, status=status, ms=round((time.time() - began) * 1000.0, 1))
        with self._lock:
            self.requests.append(record)
        h._send(status, payload, raw=raw_out)

    def bodies(self, ok_only: bool = True) -> List[Dict[str, Any]]:
        return [r["body"] for r in self.requests if isinstance(r.get("body"), dict) and (r["status"] == 200 or not ok_only)]
