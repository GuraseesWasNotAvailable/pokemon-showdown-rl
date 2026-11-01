"""
Bridge between poke-env and the PSBot heuristic.

PSBot is the JavaScript bot in psbot/psbot.js. It normally runs inside the Showdown web
client. heuristic/engine.js runs that same file under Node, so this module can ask it for a
move from Python: the battle protocol lines a poke-env player receives are forwarded to the
engine, and the engine answers with its choice and a score for every legal option.

Nothing here imports torch or ray, so it is safe to use from rollout workers.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import itertools
import json
import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
ENGINE_JS = REPO_ROOT / "heuristic" / "engine.js"
SETUP_JS = REPO_ROOT / "heuristic" / "setup.js"

# Modifiers PSBot can append to a move choice.
_MOVE_FLAGS = {
    "terastallize": "terastallize",
    "mega": "mega",
    "megax": "mega",
    "megay": "mega",
    "dynamax": "dynamax",
    "zmove": "z_move",
}


class PSBotEngineError(RuntimeError):
    pass


@dataclass
class Option:
    """One legal option as PSBot scored it."""
    choice: str            # Showdown choice string, e.g. "move 2 terastallize" or "switch 4"
    label: str             # human-readable, e.g. "Earthquake" or "switch Garchomp"
    value: float           # expected material swing, in fractions of a Pokemon
    ko: Optional[float] = None    # chance this option takes a KO this turn
    dead: Optional[float] = None  # chance the user is KO'd this turn
    is_switch: bool = False


@dataclass
class Advice:
    """PSBot's answer for one request."""
    choice: str
    why: str = ""
    options: List[Option] = field(default_factory=list)
    turn: int = 0


class PSBotEngine:
    """
    One Node process running heuristic/engine.js.

    Requests are written from any thread and answered through futures, so the engine can be
    shared by every player in the process regardless of which event loop they run on.
    """

    def __init__(self, node: Optional[str] = None, engine_js: Optional[Path] = None, auto_setup: bool = True,
                 bot_js: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        self.node = node or os.environ.get("PSBOT_NODE") or shutil.which("node")
        if not self.node:
            raise PSBotEngineError("Node.js was not found. Install it or set PSBOT_NODE to the node binary.")
        self.engine_js = Path(engine_js or ENGINE_JS)
        self.bot_js = Path(bot_js) if bot_js else None  # another build of psbot.js, for A/B runs
        self.config = config  # PSBot settings for this engine only (see cfg in psbot.js)
        self.auto_setup = auto_setup
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: Dict[int, concurrent.futures.Future] = {}
        self._stderr_tail: List[str] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "PSBotEngine":
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return self
            if self.auto_setup:
                self._ensure_cache()
            env = dict(os.environ)
            if self.bot_js:
                env["PSBOT_JS"] = str(self.bot_js)
            if self.config is not None:
                env["PSBOT_CONFIG"] = json.dumps(self.config)
            self._proc = subprocess.Popen(
                [self.node, str(self.engine_js)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(REPO_ROOT), text=True, encoding="utf-8", bufsize=1, env=env,
            )
            threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
            threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()
        pong = self.call_sync({"op": "ping"}, timeout=60.0)
        logger.info("PSBot engine %s ready (pid %s)", pong.get("version"), self._proc.pid)
        return self

    def _ensure_cache(self) -> None:
        """Downloads the Showdown client files the engine needs, the first time only."""
        cache = Path(os.environ.get("PSBOT_CACHE") or (REPO_ROOT / "heuristic" / "cache"))
        if (cache / "battle.js").exists():
            return
        logger.info("Fetching Showdown client files for the PSBot engine...")
        res = subprocess.run([self.node, str(SETUP_JS)], cwd=str(REPO_ROOT), capture_output=True, text=True)
        if res.returncode != 0:
            raise PSBotEngineError(f"heuristic/setup.js failed:\n{res.stdout}\n{res.stderr}")

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        self._fail_pending("engine closed")

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- io ----------------------------------------------------------------

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            fut = self._pending.pop(msg.get("id"), None)
            if fut is None or fut.done():
                continue
            if msg.get("ok"):
                fut.set_result(msg)
            else:
                fut.set_exception(PSBotEngineError(msg.get("error") or "engine error"))
        self._fail_pending("engine exited: " + " | ".join(self._stderr_tail[-3:]))

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self._stderr_tail = (self._stderr_tail + [line])[-20:]
                logger.debug("engine: %s", line)

    def _fail_pending(self, reason: str) -> None:
        pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(PSBotEngineError(reason))

    def _write(self, msg: Dict[str, Any]) -> None:
        if not self.alive:
            self.start()
        data = json.dumps(msg, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                self._proc.stdin.write(data)
                self._proc.stdin.flush()
            except (OSError, ValueError) as e:
                raise PSBotEngineError(f"engine pipe closed: {e}") from e

    def send(self, msg: Dict[str, Any]) -> None:
        """Fire and forget. The engine handles messages in order, so no reply is needed."""
        self._write(msg)

    def request(self, msg: Dict[str, Any]) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        rid = next(self._ids)
        self._pending[rid] = fut
        try:
            self._write(dict(msg, id=rid))
        except Exception as e:
            self._pending.pop(rid, None)
            fut.set_exception(e)
        return fut

    def call_sync(self, msg: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
        return self.request(msg).result(timeout=timeout)

    async def call(self, msg: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
        return await asyncio.wait_for(asyncio.wrap_future(self.request(msg)), timeout=timeout)


_shared: Optional[PSBotEngine] = None
_shared_lock = threading.Lock()


def shared_engine() -> PSBotEngine:
    """The process-wide engine, started on first use."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = PSBotEngine()
            atexit.register(_shared.close)
        if not _shared.alive:
            _shared.start()
        return _shared


# ---------------------------------------------------------------------------
# Translating between PSBot choice strings and poke-env orders
# ---------------------------------------------------------------------------

def choice_to_order(player: Any, battle: Any, choice: str) -> Optional[Any]:
    """Turns "move 2 terastallize" or "switch 4" into a poke-env order, or None if it is not legal."""
    request = getattr(battle, "last_request", None) or {}
    parts = (choice or "").split()
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    idx = int(parts[1]) - 1
    try:
        if parts[0] == "move":
            wanted = request["active"][0]["moves"][idx]["id"]
            flags = {_MOVE_FLAGS[p]: True for p in parts[2:] if p in _MOVE_FLAGS}
            for move in battle.available_moves:
                if move.id == wanted:
                    return player.create_order(move, **flags)
            # Struggle, Recharge and friends appear under their own id in the request.
            if len(battle.available_moves) == 1:
                return player.create_order(battle.available_moves[0], **flags)
        elif parts[0] == "switch":
            ident = request["side"]["pokemon"][idx]["ident"]
            mon = battle.team.get(ident)
            if mon is not None and mon in battle.available_switches:
                return player.create_order(mon)
    except (KeyError, IndexError, TypeError):
        return None
    return None


def order_to_choice(battle: Any, order: Any) -> Optional[str]:
    """The reverse of choice_to_order, used to tell the engine what was actually played."""
    request = getattr(battle, "last_request", None) or {}
    target = getattr(order, "order", None)
    if target is None:
        return None
    try:
        if hasattr(target, "current_hp"):  # a Pokemon: this is a switch
            for i, p in enumerate(request["side"]["pokemon"]):
                if battle.team.get(p["ident"]) is target:
                    return f"switch {i + 1}"
            return None
        for i, m in enumerate(request["active"][0]["moves"]):
            if m.get("id") == target.id:
                suffix = ""
                if getattr(order, "terastallize", False):
                    suffix = " terastallize"
                elif getattr(order, "mega", False):
                    suffix = " mega"
                elif getattr(order, "dynamax", False):
                    suffix = " dynamax"
                elif getattr(order, "z_move", False):
                    suffix = " zmove"
                return f"move {i + 1}{suffix}"
    except (KeyError, IndexError, TypeError):
        return None
    return None


# ---------------------------------------------------------------------------
# Player mixin
# ---------------------------------------------------------------------------

class PSBotMixin:
    """
    Gives any poke-env Player access to PSBot.

    Put it before the Player base class:  class MyPlayer(PSBotMixin, Player): ...
    Set psbot_enabled to False to skip the forwarding entirely (plain self-play, for example).
    """

    psbot_enabled: bool = True
    psbot_timeout: float = 20.0
    _psbot_engine: Optional[PSBotEngine] = None

    @property
    def psbot(self) -> PSBotEngine:
        if self._psbot_engine is None:
            self._psbot_engine = shared_engine()
        return self._psbot_engine

    def _psbot_key(self, battle_tag: str) -> str:
        # Both sides of a self-play game can live in one process, so the tag alone is not unique.
        return f"{self.username}|{battle_tag}"

    async def _handle_battle_message(self, split_messages: List[List[str]]):
        # Forward first: poke-env calls choose_move from inside this handler, and the engine
        # has to have seen the same lines by then.
        if self.psbot_enabled and split_messages and split_messages[0] and split_messages[0][0].startswith(">"):
            room = split_messages[0][0][1:].strip()
            data = "\n".join("|".join(parts) for parts in split_messages[1:])
            if data:
                try:
                    self.psbot.send({"op": "feed", "key": self._psbot_key(room), "room": room, "data": data})
                except PSBotEngineError as e:
                    logger.warning("PSBot engine unavailable: %s", e)
        await super()._handle_battle_message(split_messages)

    def _battle_finished_callback(self, battle):
        if self.psbot_enabled:
            try:
                self.psbot.send({"op": "end", "key": self._psbot_key(battle.battle_tag)})
            except PSBotEngineError:
                pass
        super()._battle_finished_callback(battle)

    async def psbot_advice(self, battle: Any) -> Advice:
        """PSBot's choice and its score for every legal option on the current request."""
        res = await self.psbot.call({
            "op": "decide",
            "key": self._psbot_key(battle.battle_tag),
            "room": battle.battle_tag,
            "request": getattr(battle, "last_request", None),
        }, timeout=self.psbot_timeout)
        options = [
            Option(choice=o["choice"], label=o.get("label") or "", value=float(o["v"] if o.get("v") is not None else 0.0),
                   ko=o.get("ko"), dead=o.get("dead"), is_switch=bool(o.get("sw")))
            for o in res.get("table") or [] if o.get("choice")
        ]
        return Advice(choice=res.get("choice") or "default", why=res.get("why") or "", options=options, turn=int(res.get("turn") or 0))

    def psbot_note(self, battle: Any, order: Any) -> None:
        """Tells the engine which order was really sent, so its switch bookkeeping stays right."""
        choice = order_to_choice(battle, order)
        if not choice:
            return
        try:
            self.psbot.send({"op": "chose", "key": self._psbot_key(battle.battle_tag), "choice": choice})
        except PSBotEngineError:
            pass

    async def psbot_order(self, battle: Any) -> Optional[Any]:
        """PSBot's move as a poke-env order, or None when the engine has no legal answer."""
        try:
            advice = await self.psbot_advice(battle)
        except Exception as e:  # engine down, timeout, bad state
            logger.warning("PSBot advice failed in %s: %s", battle.battle_tag, e)
            return None
        order = choice_to_order(self, battle, advice.choice)
        if order is not None:
            self.psbot_note(battle, order)
        return order
