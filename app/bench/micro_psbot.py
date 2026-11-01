#!/usr/bin/env python
"""
Micro-benchmark of the imitation teacher: how many decisions a second ONE PSBot engine process
(node heuristic/engine.js) can answer, fed the way worker.py feeds it.

In imitation with --teacher psbot every rollout worker has exactly one engine (psbot_bridge's
shared_engine), and every decision of both sides of every battle in that worker waits for it.
This replays recorded player streams (bench/micro_showdown_sim.py --record) into one engine
with the same three messages PSBotMixin sends: "feed" for every battle message, "decide" for
every request that needs a choice, "chose" afterwards, "end" when the battle is over.

    python bench/micro_psbot.py --streams <dir>/streams.jsonl --battles 40 --out <dir>

It starts one node process through psbot_bridge.PSBotEngine and closes it again. No Showdown
server, no Ray, no GPU. The recorded games were played by a random-move policy, so the
positions are a little wilder than a trained policy's, but the engine's work per request (its
damage calculations over every option) is the same kind.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--battles", type=int, default=40)
    ap.add_argument("--warm-battles", type=int, default=4, help="battles replayed first and not counted (JIT, set data)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.streams = os.path.abspath(args.streams)
    if args.out:
        args.out = os.path.abspath(args.out)
        os.makedirs(args.out, exist_ok=True)
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)
    from psbot_bridge import PSBotEngine

    msgs: List[Dict[str, Any]] = []
    with open(args.streams, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["b"] >= args.battles + args.warm_battles:
                break
            msgs.append(d)

    eng = PSBotEngine()
    t0 = time.perf_counter()
    eng.start()
    start_s = time.perf_counter() - t0
    names = {"p1": "benchpa", "p2": "benchpb"}
    rq: Dict[str, int] = {}
    lat: List[float] = []
    n_feed = n_decide = n_wait = n_fail = 0
    bytes_out = 0
    py_cpu0 = t_count0 = None
    counted_from = args.warm_battles
    choices: Dict[str, int] = {}
    try:
        for d in msgs:
            if d["b"] == counted_from and t_count0 is None:
                eng.call_sync({"op": "ping"}, timeout=120.0)      # everything fed so far is processed
                t_count0 = time.perf_counter()
                py_cpu0 = time.process_time()
                lat.clear()
                choices.clear()
                n_feed = n_decide = n_wait = n_fail = 0
                bytes_out = 0
            m = d["m"]
            nl = m.find("\n")
            room = m[1:nl].strip()
            body = m[nl + 1:]
            key = f"{names[d['p']]}|{room}"
            eng.send({"op": "feed", "key": key, "room": room, "data": body})
            n_feed += 1
            bytes_out += len(body)
            if body.startswith("|request|"):
                req = json.loads(body[9:])
                rq[key] = rq.get(key, 0) + 1
                req["rqid"] = rq[key]
                if req.get("wait"):
                    n_wait += 1
                else:
                    t1 = time.perf_counter()
                    try:
                        res = eng.call_sync({"op": "decide", "key": key, "room": room, "request": req}, timeout=60.0)
                        choice = res.get("choice") or "default"
                    except Exception:  # noqa: BLE001
                        n_fail += 1
                        choice = "default"
                    lat.append((time.perf_counter() - t1) * 1000.0)
                    n_decide += 1
                    kind = choice.split(" ")[0]
                    choices[kind] = choices.get(kind, 0) + 1
                    eng.send({"op": "chose", "key": key, "choice": choice})
                    bytes_out += len(body)
            if "|win|" in body or "\n|tie" in body:
                eng.send({"op": "end", "key": key})
        eng.call_sync({"op": "ping"}, timeout=120.0)
        wall = time.perf_counter() - (t_count0 or t0)
        py_cpu = time.process_time() - (py_cpu0 or 0.0)
    finally:
        eng.close()

    lat.sort()
    res = {
        "battles_counted": args.battles, "engine_start_seconds": round(start_s, 2),
        "decisions": n_decide, "wait_requests": n_wait, "feeds": n_feed, "failed_decides": n_fail,
        "wall_seconds": round(wall, 2),
        "engine_ms_per_decision_including_its_feeds": round(wall * 1000.0 / max(n_decide, 1), 3),
        "decisions_per_second_one_engine": round(n_decide / wall, 1),
        "decide_round_trip_ms_median": round(statistics.median(lat), 3) if lat else None,
        "decide_round_trip_ms_p90": round(lat[int(0.9 * len(lat))], 3) if lat else None,
        "decide_round_trip_ms_max": round(lat[-1], 1) if lat else None,
        "python_side_cpu_ms_per_decision": round(py_cpu * 1000.0 / max(n_decide, 1), 3),
        "kilobytes_sent_per_decision": round(bytes_out / 1024.0 / max(n_decide, 1), 2),
        "choice_kinds": choices,
    }
    if args.out:
        with open(os.path.join(args.out, "psbot_bench.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
