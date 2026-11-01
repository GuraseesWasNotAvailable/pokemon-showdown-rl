"""
Does the bridge's helper keep a battle exactly as Foul Play itself keeps it?

    python bench/fp_bridge_replay.py check                  # exit code 0: it does

An exact check. No search is run and no server is needed, it takes about 20 seconds, and it
answers the one question that win rates and "same move" percentages cannot: whether
bench/fp_bridge_server.py tracks every battle the way Foul Play's own unmodified battle loop
does, line for line. A head to head of 200 games cannot see a helper that is wrong in one
position out of fifty. This sees a single line that was not fed, or a single choice that was
noted wrongly, at the first decision where Foul Play knows something else because of it.

How. Recorded battles (tests/fixtures/foulplay_bridge_replay.jsonl.gz: the messages a real Foul
Play process received in three of its games, and those of a poke-env player in games of random
moves, with Transform, Illusion, locked moves, recharge turns and the like) are replayed twice,
each time in a fresh process of FOUL PLAY's virtualenv python:

  pristine   fp.run_battle.pokemon_battle, unmodified, is fed the messages. Where the recorded
             player moved, Foul Play's own async_pick_move runs: copy the battle, update it
             from the request, sample the opponent's teams, turn each into an engine state.
  bridge     bench/fp_bridge_server.py is fed the same messages the way foulplay_bridge.py
             feeds it: feed, decide, then chose with the move the player really sent.

In both the engine call is replaced by a stub that keeps the state string it was given, the
move "chosen" is the move the recorded player sent, and Python's random is seeded per decision.
With the same battle state the sampled teams are then identical, so the state strings of every
sampled world must be EQUAL, byte for byte, at every decision. Compared at each one:

  states     the engine state string of every sampled world (2 or 4)
  ms         the search time Foul Play gives each world
  picture    everything in Foul Play's Battle object, also what the engine never sees
  message    what Foul Play would send for the move
  lsm        last_selected_move, which Foul Play's inference about the opponent starts from:
             the helper's "chose" has to set it, alone, to what Foul Play's own code sets

and for the bridge by itself: a battle rebuilt from its stored lines and choices (what happens
after a helper died) and all battles interleaved in one process (what a helper really holds)
must give the same states as the battle followed alone.

    python bench/fp_bridge_replay.py check [fixture] [--foulplay-dir DIR] [--mutate KIND]
    python bench/fp_bridge_replay.py make OUT.jsonl.gz [--fp-log LOG[:n,n]]... [--trace FILE]...

--mutate breaks the bridge's input on purpose (drop-line: one line of one message, a move of
the opponent, is not fed; wrong-note: one "chose" names another move than the one sent) to
show that the check notices. The check must then FAIL. tests/test_foulplay_bridge.py runs all
three.

Both processes get the same PYTHONHASHSEED: Foul Play's own tracking depends on the order of a
set in one rare place, so two processes with different hash seeds can differ by themselves.

Nothing of Foul Play and nothing of this repository is written to. Foul Play is found in
--foulplay-dir, $FOULPLAY_DIR or ~/foul-play, as everywhere else.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import types
import zlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_FIXTURE = os.path.join(REPO, "tests", "fixtures", "foulplay_bridge_replay.jsonl.gz")
BATTLE_FORMAT = "gen9randombattle"
MUTATIONS = ("drop-line", "wrong-note")
FIELDS = ("rqid", "states", "ms", "picture", "message", "lsm")
REBUILD_EVERY = 5      # every fifth decision is asked again in a battle rebuilt from the stored lines and choices


# =============================================================================================
# The fixture
# =============================================================================================

def load_fixture(path: str) -> List[Dict[str, Any]]:
    """Battles: {"name", "user", "room", "source", "steps": [{"data", "decision": None or {"rqid", "slot"}}]}."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _to_id(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _slot_of(message: str, request: Dict[str, Any]) -> Optional[str]:
    """What a player sent ("/choose move earthquake terastallize", "/switch 3", "/choose switch garchomp") with slot numbers."""
    parts = message.split()
    if len(parts) == 2 and parts[0] == "/switch" and parts[1].isdigit():
        return "switch %s" % parts[1]
    if parts[:2] == ["/choose", "move"] and len(parts) >= 3:
        active = request.get("active") or []
        for i, move in enumerate((active[0].get("moves") or []) if active else []):
            if move.get("id") == parts[2] or (parts[2].isdigit() and int(parts[2]) == i + 1):
                return "move %d%s" % (i + 1, " terastallize" if "terastallize" in parts[3:] else "")
        return None
    if parts[:2] == ["/choose", "switch"] and len(parts) >= 3:
        if parts[2].isdigit():
            return "switch %s" % parts[2]
        name = _to_id(" ".join(parts[2:]))
        for i, mon in enumerate((request.get("side") or {}).get("pokemon") or []):
            if _to_id(mon["details"].split(",")[0]) == name or _to_id(mon["ident"].split(":", 1)[1]) == name:
                return "switch %d" % (i + 1)
    return None


def _last_request(data: str, before: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for line in data.split("\n"):
        if line.startswith("|request|") and len(line) > 9:
            try:
                before = json.loads(line[9:])
            except ValueError:
                pass
    return before


def battles_of_foulplay_log(path: str) -> List[Dict[str, Any]]:
    """The battles of one real Foul Play process, from its DEBUG log: every message it received, every move it sent."""
    received, sent = "Received message from websocket: ", "Sending message to websocket: "
    levels = {level.ljust(8) + " " for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")}
    records: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line[:9] in levels:
                records.append(line[9:])
            elif records:
                records[-1] += "\n" + line
    user = None
    for text in records:
        found = re.match(r"^Logged in as (\S+) without an assertion", text)
        if found:
            user = found.group(1)
            break
    battles: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    request: Optional[Dict[str, Any]] = None
    for text in records:
        if text.startswith(received):
            message = text[len(received):]
            if not message.startswith(">battle-"):
                continue                                   # the bridge is fed battle rooms only
            head, _, data = message.partition("\n")
            room = head[1:].strip()
            if data.startswith("|init|battle"):
                current = {"name": "%s_%s" % (os.path.basename(path).rsplit(".", 1)[0], room), "user": user, "room": room,
                           "source": "a real Foul Play process (its DEBUG log)", "steps": []}
                battles.append(current)
                request = None
            if current is None or room != current["room"] or not data:
                continue
            current["steps"].append({"data": data, "decision": None})
            request = _last_request(data, request)
        elif text.startswith(sent) and current is not None and current["steps"]:
            bits = text[len(sent):].split("|")
            if len(bits) < 3 or bits[0] != current["room"] or not (bits[1].startswith("/choose move") or bits[1].startswith("/switch")):
                continue
            slot = _slot_of(bits[1], request or {})
            if current["steps"][-1]["decision"] is None:
                current["steps"][-1]["decision"] = {"rqid": str(bits[2]), "slot": slot, "sent": bits[1]}
    return battles


def battle_of_trace(path: str) -> Dict[str, Any]:
    """One battle as a poke-env player received it (one JSON line per event: "msg", "order", "end")."""
    with open(path, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]
    user = events[-1].get("username")
    room = next(e["room"] for e in events if e["t"] == "msg")
    steps: List[Dict[str, Any]] = []
    for event in events:
        if event["t"] == "msg":
            steps.append({"data": event["data"], "decision": None})
        elif event["t"] == "order" and steps and steps[-1]["decision"] is None:
            request = event.get("request") or {}
            steps[-1]["decision"] = {"rqid": str(request.get("rqid")), "slot": _slot_of(event["message"], request), "sent": event["message"],
                                     "reviving": any(p.get("reviving") for p in (request.get("side") or {}).get("pokemon") or [])}
    return {"name": os.path.basename(path).rsplit(".", 1)[0], "user": user, "room": room,
            "source": "a poke-env player (recorded websocket messages)", "steps": steps}


def make(out: str, fp_logs: List[str], traces: List[str]) -> int:
    battles: List[Dict[str, Any]] = []
    for spec in fp_logs:
        path, _, picks = spec.partition(":")
        found = battles_of_foulplay_log(path)
        wanted = [int(x) for x in picks.split(",") if x.strip()] if picks else list(range(len(found)))
        battles.extend(found[i] for i in wanted)
    battles.extend(battle_of_trace(path) for path in traces)
    kept = []
    for battle in battles:
        decisions = [s["decision"] for s in battle["steps"] if s["decision"]]
        if any(d["slot"] is None or d.get("reviving") for d in decisions):
            print("left out %s: a move that has no slot, or a Revival Blessing request" % battle["name"])
            continue
        for d in decisions:
            d.pop("reviving", None)
        kept.append(battle)
        print("%-60s %4d messages, %3d decisions (%s)" % (battle["name"], len(battle["steps"]), len(decisions), battle["source"]))
    if os.path.exists(out):
        print("%s exists: not overwritten" % out)
        return 2
    with gzip.open(out, "wt", encoding="utf-8", newline="\n") as fh:
        for battle in kept:
            fh.write(json.dumps(battle, separators=(",", ":")) + "\n")
    print("%d battles written to %s (%d bytes)" % (len(kept), out, os.path.getsize(out)))
    return 0


# =============================================================================================
# The two replays (these run under Foul Play's python)
# =============================================================================================

def _picture(obj: Any, depth: int = 0) -> Any:
    """Everything in a Foul Play Battle, in a form two processes write alike."""
    if depth > 8:
        return "..."
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted((_picture(x, depth + 1) for x in obj), key=repr)
    if isinstance(obj, dict):
        return {str(k): _picture(v, depth + 1) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
                if not (isinstance(obj, defaultdict) and v == 0)}
    if isinstance(obj, (list, tuple)):
        return [_picture(x, depth + 1) for x in obj]
    if hasattr(obj, "__dict__"):
        return {k: _picture(v, depth + 1) for k, v in sorted(vars(obj).items()) if k not in ("mode", "request_json")}
    return repr(obj)


def _md5(value: Any) -> str:
    return hashlib.md5(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _seed(name: str, rqid: Any, n: int) -> int:
    return zlib.crc32(("%s|%s|%d" % (name, rqid, n)).encode("utf-8"))


class _Capture:
    """What the stubs keep of one decision."""

    def __init__(self) -> None:
        self.slot: Optional[str] = None
        self.seed = 0
        self.keep_text = False
        self.reset()

    def reset(self) -> None:
        self.states: List[str] = []
        self.ms: List[int] = []
        self.picture: Any = None
        self.forced: Optional[str] = None

    def row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"states": [hashlib.md5(s.encode("utf-8")).hexdigest() for s in self.states], "ms": list(self.ms),
                               "picture": _md5(self.picture), "forced": self.forced}
        if self.keep_text:
            row["states_text"] = list(self.states)
            row["picture_text"] = self.picture
        return row


CAP = _Capture()


def _forced_name(battle: Any, slot: Optional[str]) -> Optional[str]:
    """The recorded choice in Foul Play's own words, read off the battle Foul Play prepared for this decision."""
    parts = (slot or "").split()
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    n = int(parts[1])
    if parts[0] == "switch":
        for mon in battle.user.reserve:
            if mon.index == n:
                return "switch " + mon.name
        return None
    moves = battle.request_json["active"][0]["moves"]
    if not 1 <= n <= len(moves):
        return None
    name = moves[n - 1]["id"]
    if name == "hiddenpower":
        name = _to_id(moves[n - 1]["move"])
    move = battle.user.active.get_move(name)
    return (move.name if move is not None else name) + ("-tera" if "terastallize" in parts[2:] else "")


def _install_stubs(modes_base: Any, search_main: Any) -> None:
    """No search: the engine call keeps its state string, the "search result" is the recorded move, random is seeded."""

    def no_search(state: str, search_time_ms: int, index: int, threads: int) -> Any:
        CAP.states.append(state)
        CAP.ms.append(int(search_time_ms))
        return types.SimpleNamespace(total_visits=1, side_one=[])

    def recorded_move(_results: Any) -> Optional[str]:
        return CAP.forced

    search_main.get_result_from_mcts = no_search
    search_main.select_move_from_mcts_results = recorded_move
    inner = modes_base.find_best_move

    def find_best_move(battle: Any) -> str:
        CAP.picture = _picture(battle)
        CAP.forced = _forced_name(battle, CAP.slot)
        random.seed(CAP.seed)
        return inner(battle)

    modes_base.find_best_move = find_best_move


class _InlineExecutor:
    def __init__(self, max_workers: Optional[int] = None, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        import concurrent.futures

        future: Any = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class _Exhausted(Exception):
    pass


def _enter_foul_play(fp_dir: str) -> None:
    os.chdir(fp_dir)
    sys.path.insert(0, fp_dir)


def run_pristine(fp_dir: str, battles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Foul Play's own battle loop, unmodified, fed the recorded messages."""
    _enter_foul_play(fp_dir)
    from fp.config import FoulPlayConfig

    argv, sys.argv = sys.argv, ["run.py", "--websocket-uri", "none", "--ps-username", "replay", "--bot-mode", "accept_challenge",
                                "--pokemon-format", BATTLE_FORMAT, "--search-time-ms", "100", "--search-parallelism", "1",
                                "--log-level", "WARNING"]
    try:
        FoulPlayConfig.configure()
    finally:
        sys.argv = argv
    from fp.data.mods.apply_mods import apply_mods

    apply_mods(FoulPlayConfig.format_spec)
    import fp.modes.base as modes_base
    import fp.modes.random_battle as modes_random
    import fp.run_battle as run_battle
    import fp.search.main as search_main

    search_main.ProcessPoolExecutor = _InlineExecutor      # Foul Play forks a worker per decision: the stub must run here
    _install_stubs(modes_base, search_main)
    original_pick = modes_base.async_pick_move
    seen: Dict[str, Any] = {}

    async def watched_pick(battle: Any) -> Any:
        seen["picked"] = True
        result = await original_pick(battle)                # Foul Play's own, untouched
        seen["lsm"] = list(battle.user.last_selected_move)
        return result

    modes_random.async_pick_move = watched_pick
    run_battle.async_pick_move = watched_pick

    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for battle in battles:
        name, room, steps = battle["name"], battle["room"], battle["steps"]
        FoulPlayConfig.username = battle["user"]
        at = {"i": -1, "n": 0}

        def close_step() -> None:
            i = at["i"]
            if not 0 <= i < len(steps):
                return
            decision = steps[i]["decision"]
            if decision and not seen.get("picked"):
                raise RuntimeError("the recorded player moved on message %d (request %s), Foul Play's loop did not" % (i, decision["rqid"]))
            if not decision and seen.get("picked"):
                raise RuntimeError("Foul Play's loop moved on message %d, the recorded player did not" % i)
            if decision:
                row = dict(CAP.row(), battle=name, n=at["n"], rqid=str(decision["rqid"]), slot=decision["slot"],
                           message=seen.get("message"), lsm=seen.get("lsm"))
                rows.append(row)
                at["n"] += 1

        class Replayed:
            async def receive_message(self) -> str:
                close_step()                                # the answer to the message before has been given by now
                at["i"] += 1
                if at["i"] >= len(steps):
                    raise _Exhausted()
                decision = steps[at["i"]]["decision"]
                CAP.reset()
                seen.clear()
                CAP.slot = decision["slot"] if decision else None
                if decision:
                    CAP.seed = _seed(name, decision["rqid"], at["n"])
                return ">" + room + "\n" + steps[at["i"]]["data"]

            async def send_message(self, _room: str, message_list: List[str]) -> None:
                if message_list and (message_list[0].startswith("/choose move") or message_list[0].startswith("/switch")):
                    seen["message"] = message_list[0]

            async def leave_battle(self, _tag: str) -> None:
                return None

            async def save_replay(self, _tag: str) -> None:
                return None

        try:
            asyncio.run(run_battle.pokemon_battle(Replayed(), BATTLE_FORMAT, None))
            close_step()
        except _Exhausted:
            pass
        except BaseException as exc:
            problems.append("%s, message %d of %d: %s: %s" % (name, at["i"], len(steps), type(exc).__name__, str(exc)[:240]))
    return {"rows": rows, "problems": problems}


def run_bridge(fp_dir: str, battles: List[Dict[str, Any]], mutate: Optional[str]) -> Dict[str, Any]:
    """bench/fp_bridge_server.py, fed the same messages the way foulplay_bridge.py feeds it."""
    _enter_foul_play(fp_dir)
    spec = importlib.util.spec_from_file_location("fp_bridge_server", os.path.join(HERE, "fp_bridge_server.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # type: ignore[union-attr]
    server = module._Server(fp_dir, BATTLE_FORMAT, "inline", "WARNING")
    import fp.modes.base as modes_base
    import fp.search.main as search_main
    from fp.battle.state import LastUsedMove

    _install_stubs(modes_base, search_main)                 # on top of what the helper itself wraps
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    mutated: List[str] = []

    def opponent_side(battle: Dict[str, Any]) -> str:
        """"p1" or "p2": the side the recorded player is not, for --mutate drop-line."""
        for step in battle["steps"]:
            for line in step["data"].split("\n"):
                bits = line.split("|")
                if len(bits) > 3 and bits[1] == "player" and bits[3] == battle["user"]:
                    return "p2" if bits[2] == "p1" else "p1"
        return "p2"

    def other_move(session: Any, slot: str) -> Optional[str]:
        """Another move of the same request, for --mutate wrong-note."""
        moves, _switches = server._slot_names(server._prepared(session))
        parts = slot.split()
        sent = int(parts[1]) if parts[0] == "move" else 0
        for i, name in enumerate(moves, start=1):
            if name and i != sent:
                return "move %d" % i
        return None

    def play(which: str, order: List[Tuple[Dict[str, Any], Dict[str, Any]]], rebuild: bool) -> None:
        logs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)     # what foulplay_bridge.FoulPlayBridge stores per battle
        counts: Dict[str, int] = defaultdict(int)
        for battle, step in order:
            name, user, room = battle["name"], battle["user"], battle["room"]
            key = "%s|%s|%s" % (user, which, name)
            data = step["data"]
            if mutate == "drop-line" and which == "alone" and not mutated and counts[key] >= 3:
                # One line of one message: a move of the OPPONENT's Pokemon. Foul Play then does not
                # know that move of it, and nothing that follows tells it again.
                lines = data.split("\n")
                victims = [i for i, line in enumerate(lines) if line.startswith("|move|%sa: " % opponent_side(battle))]
                if victims:
                    mutated.append("%s: not fed %r" % (name, lines[victims[0]]))
                    del lines[victims[0]]
                    data = "\n".join(lines)
            if data:
                server.handle({"op": "feed", "key": key, "user": user, "room": room, "data": data})
                logs[key].append(("feed", data))
            decision = step["decision"]
            if not decision:
                continue
            n = counts[key]
            counts[key] += 1
            CAP.reset()
            CAP.slot = decision["slot"]
            CAP.seed = _seed(name, decision["rqid"], n)
            reply = server.handle({"op": "decide", "id": 1, "key": key, "ms": 100, "rqid": decision["rqid"]})
            row = dict(CAP.row(), battle=name, n=n, rqid=str(decision["rqid"]), slot=decision["slot"], run=which)
            session = server.sessions.get(key)
            if not reply.get("ok") or session is None:
                row["error"] = str(reply.get("error"))
                problems.append("%s (%s), decision %d, request %s: %s" % (name, which, n, decision["rqid"], str(reply.get("error"))[:240]))
                rows.append(row)
                continue
            row["message"] = reply.get("fp_message")
            noted = decision["slot"]
            if mutate == "wrong-note" and which == "alone" and not mutated and n >= 3:
                wrong = other_move(session, noted)
                if wrong:
                    mutated.append("%s, decision %d: noted %r, the player sent %r" % (name, n, wrong, noted))
                    noted = wrong
            # "chose" has to set last_selected_move by itself: here it is what the player sent, but in a
            # game it is whatever was played, also when that is not the move Foul Play picked. So what
            # Foul Play's own pick left there is wiped first.
            session.battle.user.last_selected_move = LastUsedMove("", "nothing was noted", -1)
            server.handle({"op": "chose", "key": key, "choice": noted})
            logs[key].append(("chose", noted))
            row["lsm"] = list(session.battle.user.last_selected_move)
            rows.append(row)
            if rebuild and n % REBUILD_EVERY == 0:
                # What a new helper is given after the old one died: every stored line and choice
                # of the battle up to this question, and then the question again.
                again = "%s|rebuilt|%s|%d" % (user, name, n)
                for kind, payload in logs[key][:-1]:
                    if kind == "feed":
                        server.handle({"op": "feed", "key": again, "user": user, "room": room, "data": payload})
                    else:
                        server.handle({"op": "chose", "key": again, "choice": payload})
                CAP.reset()
                CAP.slot = decision["slot"]
                CAP.seed = _seed(name, decision["rqid"], n)
                reply = server.handle({"op": "decide", "id": 1, "key": again, "ms": 100, "rqid": decision["rqid"]})
                rebuilt = dict(CAP.row(), battle=name, n=n, rqid=str(decision["rqid"]), slot=decision["slot"], run="rebuilt")
                if not reply.get("ok"):
                    rebuilt["error"] = str(reply.get("error"))
                    problems.append("%s (rebuilt), decision %d: %s" % (name, n, str(reply.get("error"))[:240]))
                rows.append(rebuilt)
                server.handle({"op": "end", "key": again})

    alone = [(battle, step) for battle in battles for step in battle["steps"]]
    play("alone", alone, rebuild=True)
    mixed: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    cursors = [0] * len(battles)
    while any(c < len(b["steps"]) for c, b in zip(cursors, battles)):
        for i, battle in enumerate(battles):
            if cursors[i] < len(battle["steps"]):
                mixed.append((battle, battle["steps"][cursors[i]]))
                cursors[i] += 1
    play("mixed", mixed, rebuild=False)
    failed = [(s.key, s.error) for s in server.sessions.values() if s.state == "failed"]
    for key, error in failed:
        problems.append("%s: %s" % (key, error))
    return {"rows": rows, "problems": problems, "mutated": mutated, "counters": dict(server.counters)}


# =============================================================================================
# The comparison (this runs under any python)
# =============================================================================================

def _foulplay_python(fp_dir: str) -> Optional[str]:
    override = os.environ.get("FOULPLAY_PYTHON")
    if override:
        return override if os.path.isfile(override) else None
    for rel in ((".venv", "bin", "python"), ("venv", "bin", "python")):
        candidate = os.path.join(fp_dir, *rel)
        if os.path.isfile(candidate):
            return candidate
    return None


def _child(python: str, fp_dir: str, mode: str, fixture: str, mutate: Optional[str], keep_text: bool,
           battles: Optional[int]) -> Dict[str, Any]:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        env.pop(key, None)
    env.update({"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
    command = [python, "-u", os.path.abspath(__file__), "run", mode, fixture, "--foulplay-dir", fp_dir]
    if mutate:
        command += ["--mutate", mutate]
    if keep_text:
        command += ["--text"]
    if battles:
        command += ["--battles", str(battles)]
    done = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-12:]
        raise RuntimeError("the %s replay stopped with exit code %d:\n%s" % (mode, done.returncode, "\n".join(tail)))
    return json.loads(done.stdout.decode("utf-8"))


def _first_difference(a: Any, b: Any, path: str = "") -> str:
    if type(a) is not type(b):
        return "%s: %r vs %r" % (path, a, b)
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                return _first_difference(a.get(key), b.get(key), path + "." + str(key))
        return ""
    if isinstance(a, list):
        if len(a) != len(b):
            return "%s: %d entries vs %d" % (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _first_difference(x, y, "%s[%d]" % (path, i))
        return ""
    return "%s: %r vs %r" % (path, a, b)


def check(fixture: str, fp_dir: str, mutate: Optional[str] = None, keep_text: bool = False,
          battles: Optional[int] = None) -> Dict[str, Any]:
    """Runs both replays (of the first `battles` battles, default all) and compares them. "ok" is the verdict."""
    python = _foulplay_python(fp_dir)
    if python is None:
        raise RuntimeError("Foul Play at %s has no virtualenv python (.venv/bin/python): run bench/install_foulplay.sh" % fp_dir)
    recorded = load_fixture(fixture)[:battles or None]
    pristine = _child(python, fp_dir, "pristine", fixture, None, keep_text, battles)
    bridge = _child(python, fp_dir, "bridge", fixture, mutate, keep_text, battles)

    theirs = {(r["battle"], r["n"]): r for r in pristine["rows"]}
    ours: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {"alone": {}, "mixed": {}, "rebuilt": {}}
    for row in bridge["rows"]:
        ours[row["run"]][(row["battle"], row["n"])] = row

    differences: List[str] = []
    counts = {field: 0 for field in FIELDS}

    def differ(where: str, key: Tuple[str, int], field: str, a: Dict[str, Any], b: Dict[str, Any]) -> None:
        counts[field] += 1
        detail = ""
        if field == "states" and "states_text" in a and "states_text" in b:
            detail = " " + _first_difference(a["states_text"], b["states_text"])[:400]
        elif field == "picture" and "picture_text" in a and "picture_text" in b:
            detail = " " + _first_difference(a["picture_text"], b["picture_text"])[:400]
        elif field not in ("states", "picture"):
            detail = " %r vs %r" % (a.get(field), b.get(field))
        differences.append("%s: %s decision %d (request %s, the player sent %r): %s differs%s" % (
            where, key[0], key[1], a.get("rqid"), a.get("slot"), field, detail))

    for key, a in sorted(theirs.items()):
        b = ours["alone"].get(key)
        if b is None:
            differences.append("Foul Play's loop decided at %s decision %d, the helper was never asked there" % key)
            continue
        if "error" in b:
            continue                                        # already among the problems
        for field in FIELDS:
            if a.get(field) != b.get(field):
                differ("Foul Play's own loop vs the helper", key, field, a, b)
    for key in sorted(set(ours["alone"]) - set(theirs)):
        differences.append("the helper was asked at %s decision %d, Foul Play's loop never decided there" % key)
    for which, label, fields in (("mixed", "the battle alone vs among the others", ("states", "ms", "picture", "message", "lsm")),
                                 ("rebuilt", "the battle as followed vs rebuilt from its stored lines", ("states", "ms", "picture"))):
        for key, b in sorted(ours[which].items()):
            a = ours["alone"].get(key)
            if a is None or "error" in a or "error" in b:
                continue
            for field in fields:
                if a.get(field) != b.get(field):
                    differ(label, key, field, a, b)

    problems = ["Foul Play's own loop: " + p for p in pristine["problems"]] + ["the helper: " + p for p in bridge["problems"]]
    compared = len([k for k in theirs if k in ours["alone"]])
    return {
        "ok": not differences and not problems and compared > 0,
        "battles": len(recorded), "decisions": len(theirs), "compared": compared,
        "worlds": sum(len(r["states"]) for r in theirs.values()),
        "mixed": len(ours["mixed"]), "rebuilt": len(ours["rebuilt"]),
        "differences": len(differences), "by_field": counts, "first_differences": differences[:12],
        "problems": problems[:12], "mutated": bridge.get("mutated") or [], "counters": bridge.get("counters") or {},
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Exact check: the bridge's helper against Foul Play's own battle loop.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check", help="replay the fixture both ways and compare (exit code 0: no difference)")
    p_check.add_argument("fixture", nargs="?", default=DEFAULT_FIXTURE)
    p_check.add_argument("--foulplay-dir", default=os.environ.get("FOULPLAY_DIR") or "~/foul-play")
    p_check.add_argument("--mutate", choices=MUTATIONS, default=None, help="break the helper's input on purpose: the check must fail")
    p_check.add_argument("--text", action="store_true", help="say where two state strings differ, not only that they do")
    p_check.add_argument("--battles", type=int, default=None, help="only the first N battles of the fixture")
    p_check.add_argument("--json", action="store_true", help="print the result as one JSON line only")
    p_run = sub.add_parser("run", help="(used by check) one replay, under Foul Play's python, result as JSON on stdout")
    p_run.add_argument("mode", choices=("pristine", "bridge"))
    p_run.add_argument("fixture")
    p_run.add_argument("--foulplay-dir", required=True)
    p_run.add_argument("--mutate", choices=MUTATIONS, default=None)
    p_run.add_argument("--text", action="store_true")
    p_run.add_argument("--battles", type=int, default=None)
    p_make = sub.add_parser("make", help="write a fixture from recordings")
    p_make.add_argument("out")
    p_make.add_argument("--fp-log", action="append", default=[], help="a real Foul Play process's DEBUG log, optionally :n,n for its nth battles")
    p_make.add_argument("--trace", action="append", default=[], help="a recorded poke-env battle (.jsonl)")
    args = parser.parse_args(argv)

    if args.command == "make":
        return make(args.out, args.fp_log, args.trace)

    fp_dir = os.path.abspath(os.path.expanduser(args.foulplay_dir))
    fixture = os.path.abspath(args.fixture)
    if args.command == "run":
        # The real stdout carries the result; whatever Foul Play or the engine prints goes to stderr.
        sys.stdout.flush()
        channel = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        CAP.keep_text = bool(args.text)
        battles = load_fixture(fixture)[:args.battles or None]
        result = run_pristine(fp_dir, battles) if args.mode == "pristine" else run_bridge(fp_dir, battles, args.mutate)
        channel.write(json.dumps(result, separators=(",", ":")))
        channel.flush()
        return 0

    if not os.path.isdir(os.path.join(fp_dir, "fp")):
        print("%s is not a Foul Play checkout (bench/install_foulplay.sh installs it, FOULPLAY_DIR says where)" % fp_dir)
        return 2
    if not os.path.isfile(fixture):
        print("no fixture at %s" % fixture)
        return 2
    try:
        result = check(fixture, fp_dir, args.mutate, args.text, args.battles)
    except RuntimeError as exc:
        print(str(exc))
        return 2
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
        return 0 if result["ok"] else 1
    print("%d recorded battles, %d decisions, %d sampled worlds: Foul Play's own battle loop against the bridge's helper" % (
        result["battles"], result["decisions"], result["worlds"]))
    print("  compared at %d decisions (%s): %d differences" % (result["compared"], ", ".join(FIELDS), sum(result["by_field"].values())))
    print("  %d decisions again with all battles interleaved in one helper, %d again in a battle rebuilt from its stored lines" % (
        result["mixed"], result["rebuilt"]))
    for line in result["mutated"]:
        print("  broken on purpose: %s" % line)
    for line in result["problems"]:
        print("  problem: %s" % line)
    for line in result["first_differences"]:
        print("  DIFFERENT %s" % line)
    if result["differences"] > len(result["first_differences"]):
        print("  ... and %d more" % (result["differences"] - len(result["first_differences"])))
    print("OK: the helper keeps every battle exactly as Foul Play's own loop does" if result["ok"]
          else "FAILED: %d differences, %d problems" % (result["differences"], len(result["problems"])))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
