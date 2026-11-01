#!/usr/bin/env python
"""
A soak of worker.py's players against ONE local Showdown server, with a scripted policy
(bench/MEMORY.md; written by the review of the memory fix and kept as it was).

    python start_servers.py --dir <showdown> --ports 8815 --heap-mb 1024 --log-dir <dir>
    python bench/worker_soak.py --port 8815 --out soak.jsonl --minutes 15 --server-root-pid <pid of start_servers.py>
    python bench/worker_soak.py --worker r1 ...      release r1's worker.py from git: it floods, and stops at --cap-mb

NOT a training run: no Ray, no learner, no inference actor, no RolloutWorker, no network, no GPU.
What runs is worker.py's own SyncBridgePlayer and PokeEnvVectorWrapper (this tree's, or release
r1's from git), unchanged: the client patch, the message handling, choose_move with its
observation, the episode slots, the events. What is scripted is everything behind the event
queue: a loop that answers every STEP with a random legal action (read from the observation's
action mask), counts a DONE as an episode and a CLEANUP as a dropped battle, and gives the slot
back as the rollout loop does.

Made to happen on purpose:
  * --invalid-rate: this share of the "/choose move ..." messages is replaced by "/choose move 9",
    which the server refuses ("[Invalid choice] Can't move: ... doesn't have a move 9");
  * --slow-pair: the sides of one pair get their answers --slow-think seconds late, so their
    battles reach the five minutes at which rlspawn.ts closes a room (|expire|, |deinit).

Every --sample seconds one JSON line goes to --out: private memory of this process and of the
server's process tree, every per-battle container (memwatch.container_sizes), the counters.
The last line is the summary: memory per 1,000 episodes, containers after the spawning stopped.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import os
import queue
import random
import subprocess
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R1_COMMIT = "8227671"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", choices=("new", "r1"), default="new")
    ap.add_argument("--worker-file", default=None, help="another worker.py in the place of the tree's (mutation runs)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--rooms", type=int, default=16)
    ap.add_argument("--think", type=float, default=0.1, help="seconds the scripted policy takes for a batch of decisions")
    ap.add_argument("--slow-pair", type=int, default=2, help="index of the pair whose answers come late (-1: none)")
    ap.add_argument("--slow-think", type=float, default=10.0)
    ap.add_argument("--invalid-rate", type=float, default=0.002)
    ap.add_argument("--sample", type=float, default=30.0)
    ap.add_argument("--server-root-pid", type=int, default=0, help="the launcher whose process tree is the server")
    ap.add_argument("--cap-mb", type=float, default=1500.0, help="leave when this process has grown by this much")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    os.chdir(REPO)                                   # vocab.json is opened by a relative path
    if args.worker_file:
        path = args.worker_file
    elif args.worker == "new":
        path = os.path.join(REPO, "worker.py")
    else:
        source = subprocess.run(["git", "-C", REPO, "show", f"{R1_COMMIT}:worker.py"], capture_output=True, check=True).stdout
        path = os.path.join(tempfile.mkdtemp(prefix="worker_soak_"), "worker_r1_for_soak.py")
        with open(path, "wb") as fh:
            fh.write(source)
    spec = importlib.util.spec_from_file_location(f"worker_{args.worker}_for_soak", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    import dataclasses
    import numpy as np
    import psutil
    import poke_env
    import config
    import memwatch

    rng = random.Random(args.seed)
    base = config.RunConfig.default()
    cfg = dataclasses.replace(base, learner=dataclasses.replace(base.learner, mode="ppo"),
                              rollout=dataclasses.replace(base.rollout, rooms_per_pair=args.rooms))
    st = collections.Counter()
    requests = collections.Counter()                 # (username, room) -> requests that ask for a decision
    refused_rooms = set()                            # (username, room) that had an invalid choice
    client = poke_env.ps_client.PSClient
    real_send, patched_handle = client.send_message, client._handle_message

    async def send_message(self, message, room="", message_2=None):
        if message.startswith("/choose"):
            if message == "/choose default":
                st["defaults"] += 1
            else:
                st["choices"] += 1
                if message.startswith("/choose move") and rng.random() < args.invalid_rate:
                    st["injected"] += 1
                    message = "/choose move 9"       # there is no move 9: the server refuses it
        return await real_send(self, message, room, message_2)

    async def counted(self, message):
        st["msgs"] += 1
        if message.startswith(">battle"):
            room = message[1:message.find("\n")].strip() if "\n" in message else message[1:].strip()
            if "|request|{" in message:
                for line in message.split("\n"):
                    if line.startswith("|request|{") and '"wait":true' not in line and '"teamPreview":true' not in line:
                        requests[(self.username, room)] += 1
            if "|error|[Invalid choice]" in message:
                st["invalid"] += 1
                refused_rooms.add((self.username, room))
            if "|error|[Unavailable choice]" in message:
                st["unavailable"] += 1
            if "|expire|" in message:
                st["expire_msgs"] += 1
            if "|deinit" in message:
                st["deinit_msgs"] += 1
        return await patched_handle(self, message)

    client.send_message, client._handle_message = send_message, counted

    vec = mod.PokeEnvVectorWrapper(cfg, args.pairs, args.port)
    assembler = mod.ObservationAssembler()
    m0, m1 = assembler.offsets["action_mask"]
    deadline = time.time() + 120
    while len(vec.active_pairs) < args.pairs and time.time() < deadline:
        time.sleep(0.5)
    if len(vec.active_pairs) < args.pairs:
        print("the players did not come up")
        return 3

    class Stand:                                     # what memwatch.container_sizes reads of a worker
        pass

    class Q:
        @staticmethod
        def qsize():
            return 0

    stand = Stand()
    stand.vec_env = vec
    stand._traj = {}
    stand.learner_client = Stand()
    stand.learner_client.q = Q()
    traj = stand._traj                               # tag -> {"act": [...]}, as much of the worker's buffer as is counted
    stop = threading.Event()
    hist = collections.Counter()
    mism = []

    def is_slow(tag: str) -> bool:
        if args.slow_pair < 0:
            return False
        user = tag.split("_", 1)[0]
        try:
            return int(user[-4:-1]) == args.slow_pair
        except ValueError:
            return False

    def loop():
        actions, held = {}, {}
        while not stop.is_set():
            now = time.time()
            for tag in [t for t, (when, _a) in held.items() if when <= now]:
                actions[tag] = held.pop(tag)[1]
            for tag, action in actions.items():
                if tag in vec.action_futures:
                    fut, lp = vec.action_futures.pop(tag)
                    if not fut.done():
                        lp.call_soon_threadsafe(fut.set_result, action)
                else:
                    st["answers_without_a_future"] += 1
            actions = {}
            try:
                events = [vec.event_queue.get(timeout=0.2)]
            except queue.Empty:
                continue
            time.sleep(args.think)
            while True:
                try:
                    events.append(vec.event_queue.get_nowait())
                except queue.Empty:
                    break
            for kind, tag, payload, extra in events:
                if kind == "STEP":
                    legal = np.flatnonzero(np.asarray(payload[m0:m1]) > 0)
                    action = int(legal[rng.randrange(len(legal))]) if len(legal) else 0
                    traj.setdefault(tag, {"act": []})["act"].append(action)
                    st["decisions"] += 1
                    if is_slow(tag):
                        held[tag] = (time.time() + args.slow_think, action)
                    else:
                        actions[tag] = action
                elif kind == "DONE":
                    battle = extra
                    steps = len(traj.pop(tag, {"act": []})["act"])
                    user, room = tag.split("_", 1)
                    asked = requests.pop((user, room), 0)
                    st["episodes"] += 1
                    st["episode_steps"] += steps
                    st["episodes_slow" if is_slow(tag) else "episodes_fast"] += 1
                    st["won" if battle.won else ("lost" if getattr(battle, "lost", False) else "tie")] += 1
                    had_refusal = (user, room) in refused_rooms
                    refused_rooms.discard((user, room))
                    if had_refusal:
                        st["episodes_with_refusal"] += 1
                        st["extra_steps_in_refused_episodes"] += steps - asked
                    if steps != asked:
                        st["episodes_steps_not_requests"] += 1
                        hist[steps - asked] += 1
                        if len(mism) < 20:
                            mism.append({"tag": tag, "steps": steps, "requests": asked, "refusal": had_refusal, "turn": getattr(battle, "turn", None)})
                    if steps == 0:
                        st["episodes_without_steps"] += 1
                    vec.release_slot()
                elif kind == "CLEANUP":
                    steps = len(traj.pop(tag, {"act": []})["act"])
                    user, room = tag.split("_", 1)
                    requests.pop((user, room), None)
                    refused_rooms.discard((user, room))
                    st["cleanups"] += 1
                    st["cleanup_steps"] += steps
                    vec.release_slot()
                else:
                    st["other_events"] += 1

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

    me = psutil.Process()

    def private_mb(proc) -> float:
        info = proc.memory_info()
        return getattr(info, "private", info.rss) / 1e6

    def server_mb() -> float:
        if not args.server_root_pid:
            return 0.0
        total = 0.0
        try:
            for p in psutil.Process(args.server_root_pid).children(recursive=True):
                try:
                    total += private_mb(p)
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        return total

    def players():
        return [p for pair in vec.active_pairs for p in pair]

    def sample(t0):
        sizes = memwatch.container_sizes(stand)
        sizes.update(getattr(mod, "CHOICE_REPAIR", {}))
        sizes["default_marks"] = sum(len(p.ps_client.__dict__.get("_default_sent", ())) for p in players())
        sizes["requests_dict"] = len(requests)
        sizes["ep_sem_value"] = vec.ep_sem._value
        row = {"up": round(time.time() - t0, 1), "private_mb": round(private_mb(me), 1), "server_mb": round(server_mb(), 1),
               "cpu_s": round(sum(me.cpu_times()[:2]), 1), "sizes": sizes}
        row.update(st)
        row["rooms_closed"] = sum(getattr(p, "rooms_closed", 0) for p in players())
        row["choices_refused"] = sum(getattr(p, "choices_refused", 0) for p in players())
        return row

    def write(row):
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    t0 = time.time()
    rows = []
    start_mb = private_mb(me)
    reason = "time"
    while time.time() - t0 < args.minutes * 60:
        time.sleep(args.sample)
        row = sample(t0)
        rows.append(row)
        write(row)
        if not thread.is_alive():
            reason = "the scripted loop died"
            break
        if row["private_mb"] - start_mb > args.cap_mb:
            reason = "memory cap"
            break

    # Stop the spawning, let the fast battles end, and count what is left.
    async def autooff():
        for a, b in vec.active_pairs:
            try:
                await mod.safe_send(a, f"/rlautooff {a.username}, {b.username}, {cfg.env.battle_format}", room="lobby")
            except Exception as e:      # noqa: BLE001
                print("autooff:", e)

    try:
        asyncio.run_coroutine_threadsafe(autooff(), vec.async_loop).result(30)
    except Exception as e:              # noqa: BLE001
        print("autooff failed:", e)
    time.sleep(40 if reason == "time" else 5)
    end = sample(t0)
    in_flight = {}
    for i, (a, b) in enumerate(vec.active_pairs):
        in_flight[i] = [len(a._battles), len(b._battles)]
    warm = [r for r in rows if r["up"] >= 180] or rows
    first, last = warm[0], warm[-1]
    d_eps = max(1, last["episodes"] - first["episodes"])
    summary = {
        "end": reason, "worker": args.worker, "worker_file": path, "minutes": round((time.time() - t0) / 60, 1),
        "battles": args.pairs * args.rooms, "invalid_rate": args.invalid_rate,
        "episodes": end["episodes"], "decisions": end["decisions"], "steps_per_episode": round(end["episode_steps"] / max(1, end["episodes"]), 2),
        "window_s": [first["up"], last["up"]], "episodes_in_window": d_eps,
        "client_private_mb": [first["private_mb"], last["private_mb"]],
        "client_mb_per_1000_episodes": round((last["private_mb"] - first["private_mb"]) / d_eps * 1000, 3),
        "server_private_mb": [first["server_mb"], last["server_mb"]],
        "server_mb_per_1000_episodes": round((last["server_mb"] - first["server_mb"]) / d_eps * 1000, 3),
        "peak_client_tasks": max(r["sizes"].get("client_tasks", 0) for r in rows) if rows else None,
        "after_the_spawning_stopped": end["sizes"], "battles_in_flight_per_pair": in_flight,
        "counters": {k: end[k] for k in sorted(st)}, "rooms_closed": end["rooms_closed"], "choices_refused": end["choices_refused"],
        "steps_minus_requests_histogram": dict(hist), "first_mismatches": mism[:8],
    }
    write({"summary": summary})
    print(json.dumps(summary, indent=1))
    sys.stdout.flush()
    os._exit(0)                                      # poke-env's loop thread would keep the process alive


if __name__ == "__main__":
    sys.exit(main())
