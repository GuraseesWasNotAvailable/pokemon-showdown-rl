"""
Replays recorded ladder games through the Foul Play helper and keeps the WHOLE opponent side of
every answer, which is what opponent_model.py is fitted on and scored with.

    # inside WSL, from /mnt/c/Users/you/ps-ppo-search, standard library only:
    ~/psppo-bench/foul-play/.venv/bin/python bench/opp_model_replay.py --split all --allow-v1 --ms 150 --helpers 6 --tag all_ms150

bench/opp_replay.py (the scout's) does the replay itself and is used as it is: the stream of a
game (build_stream), the helper (Helper). It keeps one summed prediction per decision. This
script keeps what the model needs besides: every option of the opponent with its share, its
score, in how many sampled worlds it was an option and whether a switch target had been shown;
the set that was drawn for its active Pokemon in each world; our own options with their share;
and, when asked for (--payoff-ms), the payoff table as the helper sent it.

Written to runs/opp_pred/model/replay_<tag>.jsonl, one line per decision asked:

    {"game", "turn", "rqid", "request": "move" | "switch", "ok", "error", "exact": the active part of the
     request was recorded (not rebuilt), "iterations", "worlds", "ms", "search_ms",
     "opp": {"active", "revealed": [...], "options": [[name, share, seen, score, worlds, visits], ...],
             "sets": [{"chance", "iterations", "moves": [...], "item", "ability", "tera_type", "terastallized", "hp", "max_hp"}]},
     "ours": [[choice, share, score], ...], "payoff": the helper's block or absent}

and a summary next to it (replay_<tag>.summary.json): games, decisions, iterations per decision
(mean, median, lowest tenth), time per decision, errors.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.opp_dataset import BATTLES_DIR, OUT_DIR, REPO  # noqa: E402
from bench.opp_replay import Helper, ReplayError, build_stream, games_index, load  # noqa: E402

MODEL_DIR = os.path.join(OUT_DIR, "model")


def compact(reply: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The helper's opponent block with nothing the model does not read."""
    block = reply.get("opponent")
    if not isinstance(block, dict):
        return None
    options = [[str(o.get("name") or ""), float(o.get("share") or 0.0), o.get("seen"), o.get("score"),
                int(o.get("worlds") or 0), int(o.get("visits") or 0)] for o in block.get("options") or []]
    sets = []
    for world in block.get("worlds") or []:
        active = world.get("active") or {}
        sets.append({"chance": world.get("chance"), "iterations": world.get("iterations"), "moves": active.get("moves") or [],
                     "item": active.get("item"), "ability": active.get("ability"), "tera_type": active.get("tera_type"),
                     "terastallized": active.get("terastallized"), "hp": active.get("hp"), "max_hp": active.get("max_hp"),
                     "options": [[str(o.get("name") or ""), float(o.get("share") or 0.0)] for o in world.get("options") or []]})
    return {"active": block.get("active"), "revealed": block.get("revealed") or [], "options": options, "sets": sets}


def replay_game(helper: Helper, game: Any, ms: int, iterations: int = 0, payoff_ms: float = 0.0, payoff_mode: str = "engine",
                payoff_every: int = 1) -> Dict[str, Any]:
    stream = build_stream(game)
    key = "%s|%s" % (stream["user"], stream["room"])
    rows: List[Dict[str, Any]] = []
    asked = 0
    for event in stream["events"]:
        if event["type"] == "chose":
            helper.send({"op": "chose", "key": key, "choice": event["choice"]})
            if rows and rows[-1]["rqid"] == event["rqid"]:
                rows[-1]["recorded_choice"] = event["choice"]
            continue
        helper.send({"op": "feed", "key": key, "user": stream["user"], "room": stream["room"], "data": event["data"]})
        if event["request"] not in ("move", "switch"):
            continue
        question: Dict[str, Any] = {"op": "decide", "key": key, "ms": ms, "rqid": event["rqid"]}
        if iterations:
            question["iterations"] = iterations
        asked += 1
        with_payoff = bool(payoff_ms) and event["request"] == "move" and asked % max(1, payoff_every) == 0
        if with_payoff:
            question["payoff_ms"] = payoff_ms
            question["payoff"] = payoff_mode
        began = time.time()
        reply = helper.ask(question)
        row = {"game": stream["game"], "turn": event["turn"], "rqid": event["rqid"], "request": event["request"],
               "exact": bool(stream["exact_active"]), "ok": bool(reply.get("ok")), "error": reply.get("error"),
               "iterations": reply.get("iterations"), "worlds": reply.get("worlds"), "ms": reply.get("ms"),
               "wall_ms": round((time.time() - began) * 1000.0, 1), "search_ms": reply.get("search_ms"),
               "best": reply.get("best"), "opp": compact(reply),
               "ours": [[o.get("choice"), o.get("policy"), o.get("score")] for o in reply.get("options") or []]}
        if with_payoff and reply.get("payoff") is not None:
            row["payoff"] = reply["payoff"]
        rows.append(row)
    helper.ask({"op": "sync"})
    helper.send({"op": "end", "key": key})
    return {"game": stream["game"], "rows": rows}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recorded games through the helper, the whole opponent side kept (run inside WSL).")
    parser.add_argument("--games", type=int, default=100000)
    parser.add_argument("--skip", type=int, default=0, help="leave out the first N games of the sorted list")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--ms", type=int, default=150)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--helpers", type=int, default=6)
    parser.add_argument("--allow-v1", action="store_true")
    parser.add_argument("--payoff-ms", type=float, default=0.0)
    parser.add_argument("--payoff-mode", choices=("engine", "full"), default="engine")
    parser.add_argument("--payoff-every", type=int, default=1, help="ask for the payoff table on every Nth decision only")
    parser.add_argument("--battles", default=BATTLES_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--foulplay-dir", default=os.environ.get("FOULPLAY_DIR") or "~/psppo-bench/foul-play")
    parser.add_argument("--python", default=os.environ.get("FOULPLAY_PYTHON") or "~/psppo-bench/foul-play/.venv/bin/python")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    if not 1 <= args.helpers <= 6:
        parser.error("at most 6 helpers")

    index = [e for e in games_index(args.out) if e["aligned"] and (e["has_req"] or args.allow_v1)]
    if args.split != "all":
        index = [e for e in index if e["split"] == args.split]
    index = sorted(index, key=lambda e: e["game"])[args.skip: args.skip + args.games]
    out_dir = os.path.join(args.out, "model")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    server = os.path.join(REPO, "bench", "fp_bridge_server.py")
    lock = threading.Lock()
    problems: List[str] = []
    hello: List[Dict[str, Any]] = []
    done: List[Dict[str, Any]] = []
    started = time.time()
    path = os.path.join(out_dir, "replay_%s.jsonl" % tag)
    out = open(path, "w", encoding="utf-8", newline="\n")

    def work(worker: int) -> None:
        log = os.path.join(out_dir, "helper_%s_%d.log" % (tag, worker))
        helper = Helper(server, os.path.expanduser(args.foulplay_dir), os.path.expanduser(args.python), log)
        with lock:
            hello.append(helper.hello)
        try:
            for entry in index[worker::args.helpers]:
                try:
                    result = replay_game(helper, load(entry["game"], args.battles, args.out), args.ms, args.iterations,
                                         args.payoff_ms, args.payoff_mode, args.payoff_every)
                    with lock:
                        for row in result["rows"]:
                            row["split"] = entry["split"]
                            out.write(json.dumps(row, separators=(",", ":")) + "\n")
                            done.append({k: row.get(k) for k in ("ok", "error", "iterations", "ms", "wall_ms", "request", "game")})
                        out.flush()
                except ReplayError as exc:
                    with lock:
                        problems.append("%s: %s" % (entry["game"], exc))
                    if helper.proc.poll() is not None:
                        helper = Helper(server, os.path.expanduser(args.foulplay_dir), os.path.expanduser(args.python), log + "b")
        finally:
            helper.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.helpers) as pool:
        for future in [pool.submit(work, w) for w in range(args.helpers)]:
            future.result()
    out.close()

    answered = [d for d in done if d["ok"]]
    its = sorted(int(d["iterations"]) for d in answered if d.get("iterations"))
    walls = sorted(float(d["wall_ms"]) for d in answered)
    summary = {
        "tag": tag, "file": path, "helper": hello[0] if hello else None, "ms": args.ms, "iterations_asked": args.iterations,
        "helpers": args.helpers, "payoff_ms": args.payoff_ms, "payoff_mode": args.payoff_mode if args.payoff_ms else None,
        "games_asked": len(index), "games_replayed": len({d["game"] for d in done}), "games_not_built": problems,
        "decisions_asked": len(done), "decisions_answered": len(answered),
        "errors": dict(Counter((d["error"] or "")[:90] for d in done if not d["ok"]).most_common(12)),
        "iterations_per_decision_mean": round(sum(its) / len(its), 1) if its else None,
        "iterations_per_decision_median": its[len(its) // 2] if its else None,
        "iterations_per_decision_lowest_tenth": its[len(its) // 10] if its else None,
        "wall_ms_per_decision_mean": round(sum(walls) / len(walls), 1) if walls else None,
        "wall_ms_per_decision_p95": walls[int(0.95 * (len(walls) - 1))] if walls else None,
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "replay_%s.summary.json" % tag), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
