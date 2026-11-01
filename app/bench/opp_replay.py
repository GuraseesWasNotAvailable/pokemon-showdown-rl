"""
A recorded ladder game as the stream of websocket messages our side received, and a replay of it
through the Foul Play helper (bench/fp_bridge_server.py) so that the search answers every turn.

    from bench.opp_replay import load, messages_up_to, events_up_to, build_stream

    game = load("g1a2b3c4d5e6f")              # a game hash of runs/opp_pred/games.jsonl, or a path
    messages_up_to(game, 7)                   # every message up to and including the request of turn 7
    events_up_to(game, 7)                     # the same with the "chose" notes the helper needs between them

    # inside WSL, with any Python 3 (standard library only):
    python bench/opp_replay.py replay --split test --games 24 --ms 50 --helpers 3

WHAT THE RECORDINGS HOLD, AND WHAT IS REBUILT. The recordings do NOT hold the websocket messages
and do NOT hold the request JSON. They hold the battle log as a flat list of lines (no "|request|"
line in any of the 426 games), our team as the first request listed it (myTeam: stats, moves, item,
ability, Tera type), and per decision the request id, our team's order and conditions (mine), and,
in the 288 games of recording version 2, the active part of the request (req: move ids, pp,
disabled, trapped, maybeTrapped, canTera). So the stream is REBUILT:

  * messages: the server writes a "|t:|" line each time it resolves choices, so the log is cut
    there, and the battle's opening is cut the way the server sends it (title and joins, the
    "|player|" lines by themselves, the rest up to "|turn|1");
  * requests: after every batch but the last one. The request ids count up by two per batch
    (batch k is rqid 2k+2 for p1, 2k+3 for p2), which is how a recorded decision finds its batch.
    A batch with a recorded decision gets a move request or a forced switch request built from
    myTeam + mine + req, a batch without one gets {"wait": true};
  * our item and Terastallization are followed in the log (they are public).

Not exact: pp and disabled moves in the 138 version-1 games (no req: every move offered, full
pp), abilities changed in battle, the stats of a transformed Pokemon. Timer lines are fed after
the request they followed.

NAMES. Our name becomes "replayme", the opponent's "opp" + 10 hex of a salted hash, the room a
number made from the game hash. Chat, joins, leaves, rating lines and "X forfeited" messages are
dropped. Nothing with a real name is fed to the helper or written.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.opp_dataset import (BATTLES_DIR, OUT_DIR, REPO, alignment, batch_kind, batch_rqid, game_hash, load_game,  # noqa: E402
                               name_hash, split_batches, to_id)

ME = "replayme"
_DROP = {"j", "J", "l", "L", "n", "N", "c", "c:", "chat", "raw", "html", "uhtml", "uhtmlchange", "-message", "message",
         "error", "bigerror", "debug", "unlink", "player", "deinit", "expire", "title", "init"}


class ReplayError(Exception):
    pass


# =============================================================================================
# Finding a game
# =============================================================================================

def games_index(out_dir: str = OUT_DIR) -> List[Dict[str, Any]]:
    path = os.path.join(out_dir, "games.jsonl")
    if not os.path.isfile(path):
        raise ReplayError("%s is missing: run bench/opp_dataset.py first" % path)
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load(game: Any, battles_dir: str = BATTLES_DIR, out_dir: str = OUT_DIR) -> Dict[str, Any]:
    """A recording by its game hash ("g..."), by its path, or the recording itself."""
    if isinstance(game, dict):
        return game
    if os.path.isfile(str(game)):
        return load_game(str(game))
    for entry in games_index(out_dir):
        if entry["game"] == game:
            return load_game(os.path.join(battles_dir, entry["file"]))
    raise ReplayError("no recorded game %r" % (game,))


# =============================================================================================
# Our side of the request, rebuilt
# =============================================================================================

class _OurSide:
    def __init__(self, game: Dict[str, Any]) -> None:
        self.side = game["me"]["side"]
        self.team = [dict(m) for m in game.get("myTeam") or []]
        if not self.team:
            raise ReplayError("the recording has no team")
        self.keys: Dict[str, int] = {}
        for i, mon in enumerate(self.team):
            species = str(mon.get("details") or "").split(",")[0].strip()
            name = str(mon.get("ident") or "").split(":", 1)[-1].strip()
            for key in (name, species, mon.get("speciesForme"), mon.get("name"), species.split("-")[0]):
                if key and to_id(key) not in self.keys:
                    self.keys[to_id(key)] = i
        self.item = [str(m.get("item") or "") for m in self.team]
        self.tera = ["" for _ in self.team]
        self.last_mine: List[Dict[str, Any]] = [{"species": str(m.get("details") or "").split(",")[0].strip(),
                                                 "condition": m.get("condition"), "active": bool(m.get("active"))} for m in self.team]
        self.tera_used = False

    def _index(self, name: str) -> Optional[int]:
        return self.keys.get(to_id(name), self.keys.get(to_id(str(name).split("-")[0])))

    def observe(self, line: str) -> None:
        parts = line.split("|")
        if len(parts) < 4 or not parts[2].startswith(self.side):
            return
        index = self._index(parts[2].split(":", 1)[-1].strip())
        if index is None:
            return
        if parts[1] == "-enditem":
            self.item[index] = ""
        elif parts[1] == "-item":
            self.item[index] = to_id(parts[3])
        elif parts[1] == "-terastallize":
            self.tera[index] = parts[3]
            self.tera_used = True

    def side_json(self, mine: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        if mine:
            self.last_mine = mine
        pokemon = []
        for entry in self.last_mine:
            index = self._index(entry["species"])
            if index is None:
                raise ReplayError("the decision lists %r, which is not in the recorded team" % entry["species"])
            mon = self.team[index]
            details = str(mon.get("details") or "")
            if to_id(details.split(",")[0]) != to_id(entry["species"]):
                details = ", ".join([entry["species"]] + [f.strip() for f in details.split(",")[1:]])
            pokemon.append({"ident": mon.get("ident"), "details": details, "condition": entry.get("condition"),
                            "active": bool(entry.get("active")), "stats": mon.get("stats") or {}, "moves": list(mon.get("moves") or []),
                            "baseAbility": mon.get("baseAbility") or mon.get("ability") or "", "item": self.item[index],
                            "pokeball": mon.get("pokeball") or "pokeball", "ability": mon.get("ability") or mon.get("baseAbility") or "",
                            "commanding": False, "reviving": False, "teraType": mon.get("teraType") or "", "terastallized": self.tera[index]})
        return {"name": ME, "id": self.side, "pokemon": pokemon}

    def active_json(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        req = decision.get("req")
        if isinstance(req, dict) and req.get("moves"):
            moves = []
            for mv in req["moves"]:
                move = {"move": mv.get("name") or mv.get("move") or mv["id"], "id": mv["id"], "target": mv.get("target") or "normal"}
                if mv.get("pp") is not None:
                    move.update({"pp": mv["pp"], "maxpp": mv.get("maxpp", mv["pp"]), "disabled": bool(mv.get("disabled"))})
                moves.append(move)
            active: Dict[str, Any] = {"moves": moves}
            if req.get("canTera"):
                active["canTerastallize"] = req["canTera"]
            if req.get("trapped"):
                active["trapped"] = True
            if req.get("maybeTrapped"):
                active["maybeTrapped"] = True
            return active
        current = next((e for e in decision.get("mine") or [] if e.get("active")), None)
        index = self._index(current["species"]) if current else None
        if index is None:
            raise ReplayError("no active Pokemon in the decision")
        mon = self.team[index]
        active = {"moves": [{"move": m, "id": m, "pp": 16, "maxpp": 16, "target": "normal", "disabled": False} for m in mon.get("moves") or []]}
        if not self.tera_used and mon.get("teraType"):
            active["canTerastallize"] = mon["teraType"]
        return active


# =============================================================================================
# The stream
# =============================================================================================

def _clean(line: str, names: Dict[str, str], real: Dict[str, str]) -> Optional[str]:
    """A log line as it is fed: None when it is dropped, names replaced where the protocol has them."""
    parts = line.split("|")
    if len(parts) < 2:
        return None
    kind = parts[1]
    if kind in _DROP:
        return None
    if kind == "inactive":
        return line if len(parts) > 2 and parts[2].startswith("Time left:") else None
    if kind == "win":
        winner = "|".join(parts[2:])
        return "|win|" + next((names[s] for s in ("p1", "p2") if real[s] == winner), "somebody")
    if kind in ("-sidestart", "-sideend") and len(parts) > 2 and parts[2][:2] in names:
        parts[2] = "%s: %s" % (parts[2][:2], names[parts[2][:2]])
        return "|".join(parts)
    if kind == "-swapsideconditions":
        return "|-swapsideconditions"
    return line


def build_stream(game: Any) -> Dict[str, Any]:
    """
    {"game", "room", "user", "side", "exact_active", "events": [...]} with events in the order they are fed:
      {"type": "msg", "data": str, "turn": int, "batch": int, "rqid": None | int, "request": None | "move" | "switch" | "wait"}
      {"type": "chose", "choice": "move 2 terastallize", "rqid": int, "turn": int}
    "turn" of a request is the turn it asks a move for (a forced switch: the turn it happens in).
    """
    game = load(game)
    check = alignment(game)
    if not check["ok"]:
        raise ReplayError("the decisions do not fit the log: " + "; ".join(check["problems"][:3]))
    side = game["me"]["side"]
    other = "p2" if side == "p1" else "p1"
    gid = game_hash(game["id"])
    names = {side: ME, other: name_hash(game["opponent"]["name"])}
    real = {side: game["me"]["name"], other: game["opponent"]["name"]}
    room = "battle-gen9randombattle-%d" % int(gid[1:9], 16)
    decisions = {d["rqid"]: d for d in game.get("decisions") or []}
    ours = _OurSide(game)
    batches = split_batches(game["log"])
    events: List[Dict[str, Any]] = []
    turn = 0
    exact = True

    def msg(data: str, k: int, **more: Any) -> None:
        if data:
            events.append(dict({"type": "msg", "data": data, "turn": turn, "batch": k, "rqid": None, "request": None}, **more))

    for k, batch in enumerate(batches):
        timers = [l for l in batch if l.startswith("|inactive|Time left:")]
        core = [l for l in batch if not l.startswith("|inactive|")]
        for line in core:
            ours.observe(line)
            if line.startswith("|turn|"):
                turn = int(line.split("|")[2])
        if k == 0:
            players = [i for i, l in enumerate(core) if l.startswith("|player|")]
            stamp = next((l for l in core if l.startswith("|t:|")), "|t:|0")
            rest = [c for c in (_clean(l, names, real) for l in core[(players[-1] + 1) if players else 0:]) if c is not None]
            msg("|init|battle\n|title|%s vs. %s\n|j|☆%s\n" % (names["p1"], names["p2"], names["p1"]), k)
            msg("|j|☆%s" % names["p2"], k)
            msg(stamp + "\n|gametype|singles", k)
            msg("|player|p1|%s|1|" % names["p1"], k)
            msg("|player|p2|%s|1|\n%s" % (names["p2"], "\n".join(rest)), k)
        else:
            msg("\n".join(c for c in (_clean(l, names, real) for l in core) if c is not None), k)
        kind = batch_kind(batch)
        rqid = batch_rqid(k, side)
        decision = decisions.get(rqid)
        if kind != "end" and (decision is not None or k < len(batches) - 1):
            if decision is None:
                request: Dict[str, Any] = {"wait": True, "side": ours.side_json(None), "rqid": rqid}
                what = "wait"
            elif decision.get("forced"):
                request = {"forceSwitch": [True], "side": ours.side_json(decision.get("mine")), "rqid": rqid}
                what = "switch"
            else:
                if not isinstance(decision.get("req"), dict):
                    exact = False
                request = {"active": [ours.active_json(decision)], "side": ours.side_json(decision.get("mine")), "rqid": rqid}
                what = "move"
            msg("|request|" + json.dumps(request, separators=(",", ":")), k, rqid=rqid, request=what)
            if decision is not None and decision.get("choice"):
                events.append({"type": "chose", "choice": str(decision["choice"]), "rqid": rqid, "turn": turn})
        if timers:
            msg("\n".join(timers), k)
    return {"game": gid, "room": room, "user": ME, "side": side, "exact_active": exact, "events": events}


def events_up_to(game: Any, turn: int) -> List[Dict[str, Any]]:
    """Every event up to and including the request that asks for our move of `turn` (its own "chose" is not included)."""
    events = build_stream(game)["events"]
    for i, event in enumerate(events):
        if event["type"] == "msg" and event["request"] == "move" and event["turn"] == turn:
            return events[:i + 1]
    raise ReplayError("the game has no move request for turn %d" % turn)


def messages_up_to(game: Any, turn: int) -> List[str]:
    """
    The websocket messages of the room (the text after the ">room" line) up to and including the
    request that asks for our move of `turn`. The helper also has to be told what we sent after
    each earlier request (op "chose"): events_up_to() has those in their place.
    """
    return [event["data"] for event in events_up_to(game, turn) if event["type"] == "msg"]


# =============================================================================================
# The helper, spoken to with its documented requests only
# =============================================================================================

class Helper:
    def __init__(self, server: str, foulplay_dir: str, python: str, log_path: str) -> None:
        env = dict(os.environ, FP_BRIDGE_PARENT_PID=str(os.getpid()), PYTHONHASHSEED="0")
        self.log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen([python, "-u", server, "--foulplay-dir", foulplay_dir], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", env=env)
        self.next_id = 0
        self.hello = self.ask({"op": "ping"})

    def send(self, message: Dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def ask(self, message: Dict[str, Any]) -> Dict[str, Any]:
        self.next_id += 1
        message = dict(message, id=self.next_id)
        self.send(message)
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise ReplayError("the helper is gone (exit code %s)" % self.proc.poll())
            reply = json.loads(line)
            if reply.get("fatal"):
                raise ReplayError("the helper could not start: %s" % reply["fatal"])
            if reply.get("id") == message["id"]:
                return reply

    def close(self) -> None:
        try:
            self.send({"op": "exit"})
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        self.log.close()


def _opponent_prediction(reply: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """The helper's opponent block (protocol 2) as {"move x" | "switch y": share}, Tera folded into the move."""
    block = reply.get("opponent")
    if not isinstance(block, dict) or not block.get("options"):
        return None
    out: Dict[str, float] = {}
    for option in block["options"]:
        name = str(option.get("name") or "")
        if not name or name == "none":
            continue
        if name.endswith(" terastallize"):
            name = name[: -len(" terastallize")]
        if name.startswith("switch ") and option.get("seen") is False:
            name = "switch ?"
        out[name] = out.get(name, 0.0) + float(option.get("share") or 0.0)
    return out


def replay_game(helper: Helper, game: Any, ms: int, iterations: int = 0, ask: bool = True) -> Dict[str, Any]:
    stream = build_stream(game)
    key = "%s|%s" % (stream["user"], stream["room"])
    decisions: List[Dict[str, Any]] = []
    for event in stream["events"]:
        if event["type"] == "chose":
            helper.send({"op": "chose", "key": key, "choice": event["choice"]})
            if decisions and decisions[-1]["rqid"] == event["rqid"]:
                decisions[-1]["recorded_choice"] = event["choice"]
            continue
        helper.send({"op": "feed", "key": key, "user": stream["user"], "room": stream["room"], "data": event["data"]})
        if event["request"] in ("move", "switch") and ask:
            question = {"op": "decide", "key": key, "ms": ms, "rqid": event["rqid"]}
            if iterations:
                question["iterations"] = iterations
            reply = helper.ask(question)
            row = {"game": stream["game"], "turn": event["turn"], "rqid": event["rqid"], "request": event["request"],
                   "ok": bool(reply.get("ok")), "error": reply.get("error"), "choice": reply.get("choice"), "best": reply.get("best"),
                   "iterations": reply.get("iterations"), "worlds": reply.get("worlds"), "ms": reply.get("ms"),
                   "search_ms": reply.get("search_ms"), "revive": bool(reply.get("revive")), "failed": bool(reply.get("failed")),
                   "opp_pred": _opponent_prediction(reply),
                   "opp_active": (reply.get("opponent") or {}).get("active") if isinstance(reply.get("opponent"), dict) else None}
            decisions.append(row)
    stats = helper.ask({"op": "sync"})
    helper.send({"op": "end", "key": key})
    return {"game": stream["game"], "exact_active": stream["exact_active"], "decisions": decisions, "synced": bool(stats.get("ok"))}


def _wsl_default(path: str) -> str:
    return os.path.expanduser(path)


def run_replay(args: argparse.Namespace) -> int:
    index = [e for e in games_index(args.out) if e["aligned"] and (e["has_req"] or args.allow_v1)]
    if args.split != "all":
        index = [e for e in index if e["split"] == args.split]
    index = sorted(index, key=lambda e: e["game"])[: args.games]
    out_dir = os.path.join(args.out, "replay")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    server = os.path.join(REPO, "bench", "fp_bridge_server.py") if not args.server else args.server
    lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    problems: List[str] = []
    hello: List[Dict[str, Any]] = []
    started = time.time()

    def work(worker: int) -> None:
        helper = Helper(server, _wsl_default(args.foulplay_dir), _wsl_default(args.python), os.path.join(out_dir, "helper_%s_%d.log" % (tag, worker)))
        with lock:
            hello.append(helper.hello)
        try:
            for entry in index[worker::args.helpers]:
                try:
                    result = replay_game(helper, load(entry["game"], args.battles, args.out), args.ms, args.iterations)
                    result["split"] = entry["split"]
                    with lock:
                        results.append(result)
                except ReplayError as exc:
                    with lock:
                        problems.append("%s: %s" % (entry["game"], exc))
                    if helper.proc.poll() is not None:
                        helper = Helper(server, _wsl_default(args.foulplay_dir), _wsl_default(args.python),
                                        os.path.join(out_dir, "helper_%s_%db.log" % (tag, worker)))
        finally:
            helper.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.helpers) as pool:
        for future in [pool.submit(work, w) for w in range(args.helpers)]:
            future.result()

    asked = [d for r in results for d in r["decisions"]]
    answered = [d for d in asked if d["ok"]]
    errors = Counter((d["error"] or "")[:90] for d in asked if not d["ok"])
    its = sorted(d["iterations"] for d in answered if d.get("iterations"))
    moves = [d for d in answered if d["request"] == "move"]
    same = [d for d in answered if d.get("recorded_choice")]
    summary = {
        "tag": tag, "helper": hello[0] if hello else None, "server_file": server, "ms": args.ms, "iterations_asked": args.iterations,
        "helpers": args.helpers, "games_asked": len(index), "games_replayed": len(results),
        "games_every_decision_answered": sum(1 for r in results if r["decisions"] and all(d["ok"] for d in r["decisions"])),
        "games_not_built": problems, "decisions_asked": len(asked), "decisions_answered": len(answered),
        "move_requests_answered": len(moves), "forced_switch_requests_answered": len(answered) - len(moves),
        "errors": dict(errors.most_common(12)),
        "iterations_per_decision_mean": round(sum(its) / len(its), 1) if its else None,
        "iterations_per_decision_median": its[len(its) // 2] if its else None,
        "helper_ms_per_decision_mean": round(sum(d["ms"] for d in answered if d.get("ms")) / max(1, len(answered)), 1),
        "search_agrees_with_the_old_bots_move": round(sum(1 for d in same if d["best"] == d["recorded_choice"]) / max(1, len(same)), 4),
        "decisions_with_an_opponent_prediction": sum(1 for d in answered if d.get("opp_pred")),
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "decisions_%s.jsonl" % tag), "w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            for d in r["decisions"]:
                fh.write(json.dumps(d, separators=(",", ":")) + "\n")
    with open(os.path.join(out_dir, "summary_%s.json" % tag), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recorded games as message streams, and their replay through the Foul Play helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("replay", help="feed recorded games to the helper and ask it at every request (run inside WSL)")
    rep.add_argument("--games", type=int, default=24)
    rep.add_argument("--split", choices=("train", "test", "all"), default="test")
    rep.add_argument("--ms", type=int, default=50)
    rep.add_argument("--iterations", type=int, default=0, help="a fixed number of iterations per world, if the helper supports it")
    rep.add_argument("--helpers", type=int, default=3)
    rep.add_argument("--allow-v1", action="store_true", help="also games without the recorded active part of the request")
    rep.add_argument("--battles", default=BATTLES_DIR)
    rep.add_argument("--out", default=OUT_DIR)
    rep.add_argument("--server", default="")
    rep.add_argument("--foulplay-dir", default=os.environ.get("FOULPLAY_DIR") or "~/psppo-bench/foul-play")
    rep.add_argument("--python", default=os.environ.get("FOULPLAY_PYTHON") or "~/psppo-bench/foul-play/.venv/bin/python")
    rep.add_argument("--tag", default="")
    chk = sub.add_parser("check", help="build the stream of every game and count what cannot be built (no helper needed)")
    chk.add_argument("--battles", default=BATTLES_DIR)
    chk.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.cmd == "replay":
        if not 1 <= args.helpers <= 6:
            parser.error("at most 6 helpers")
        return run_replay(args)
    built, failed, exact, requests = 0, Counter(), 0, Counter()
    for entry in games_index(args.out):
        try:
            stream = build_stream(load(entry["game"], args.battles, args.out))
            built += 1
            exact += 1 if stream["exact_active"] else 0
            requests.update(e["request"] for e in stream["events"] if e["type"] == "msg" and e["request"])
        except ReplayError as exc:
            failed[str(exc).split(":")[0][:70]] += 1
    print(json.dumps({"games": built + sum(failed.values()), "streams_built": built, "with_the_recorded_active_part": exact,
                      "requests": dict(requests), "not_built": dict(failed)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
