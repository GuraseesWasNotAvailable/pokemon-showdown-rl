#!/usr/bin/env python
"""
One refused choice, and what a rollout worker's players do after it (bench/MEMORY.md).

Scripted players against ONE local Showdown server: no learner, no inference actor, no Ray run,
no RolloutWorker. The two players are worker.py's SyncBridgePlayer (this tree's, or release
r1's taken from git), so the module's patch of poke-env's message handler and the player's own
message handling are the real ones. What decides a move is scripted: wait --think-a (--think-b)
seconds, then a random legal move. One choice of side "a" is made invalid, once, --after
seconds in. One JSON line a second goes to --out: memory, handler tasks alive and where they
wait, messages sent and received.

    python start_servers.py --dir <showdown> --ports 8811 --log-dir <dir>      (in another shell)
    python bench/choice_flood.py --worker r1  --port 8811 --out r1.jsonl
    python bench/choice_flood.py --worker new --port 8811 --out new.jsonl
    python bench/choice_flood.py --report r1.jsonl new.jsonl

With the defaults (side "a" takes a second to decide, side "b" 0.05 s, so "a" falls behind the
server after its refused choice) release r1 ends in the flood: 43,122 refusals in 16 seconds on
the laptop. It leaves by itself when its own memory passes --cap-mb.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import os
import queue
import secrets
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R1_COMMIT = "8227671"


def report(paths) -> int:
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        rows = [r for r in rows if "private_mb" in r]
        if not rows:
            print(f"{path}: empty")
            continue
        inj = next((r["injected"] for r in rows if r.get("injected")), None)
        before = [r for r in rows if not r.get("injected")] or rows[:1]
        peak = max(rows, key=lambda r: r["private_mb"])
        flood = [r for r in rows if r["d_errors"] > 100]
        when = f"flood from {flood[0]['up'] - 1:.0f} to {flood[-1]['up']:.0f} s" if flood else "no flood"
        print(f"{path}: refused choice at {inj['at'] if inj else 'never'} s; refusals {rows[-1]['errors']:,}, "
              f"defaults sent {rows[-1]['defaults']:,}, most refusals in a second {max(r['d_errors'] for r in rows):,}; {when}; "
              f"most handler tasks at once {max(r['tasks_a'] for r in rows):,}; private memory "
              f"{max(r['private_mb'] for r in before):.0f} MB before, {peak['private_mb']:.0f} MB at the peak, "
              f"{rows[-1]['private_mb']:.0f} MB at the end; battles finished {rows[-1]['battles_done'] // 2}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", nargs="+", metavar="FILE", help="summarise files written with --out, and leave")
    ap.add_argument("--worker", choices=("new", "r1"), default="new", help="this tree's worker.py, or release r1's from git")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--out", default="choice_flood.jsonl")
    ap.add_argument("--after", type=float, default=20.0, help="seconds before the one invalid choice")
    ap.add_argument("--watch", type=float, default=120.0, help="seconds to go on after it")
    ap.add_argument("--think-a", type=float, default=1.0)
    ap.add_argument("--think-b", type=float, default=0.05)
    ap.add_argument("--cap-mb", type=float, default=3000.0)
    args = ap.parse_args()
    if args.report:
        return report(args.report)

    sys.path.insert(0, REPO)
    os.chdir(REPO)                                   # vocab.json is opened by a relative path
    if args.worker == "new":
        path = os.path.join(REPO, "worker.py")
    else:
        source = subprocess.run(["git", "-C", REPO, "show", f"{R1_COMMIT}:worker.py"], capture_output=True, check=True).stdout
        path = os.path.join(tempfile.mkdtemp(prefix="choice_flood_"), "worker_r1_for_choice_flood.py")
        with open(path, "wb") as fh:
            fh.write(source)
    spec = importlib.util.spec_from_file_location(f"worker_{args.worker}_for_choice_flood", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    import psutil
    import poke_env
    from poke_env import AccountConfiguration
    from poke_env.concurrency import POKE_LOOP
    import config

    cfg = config.RunConfig.default()
    st = dict(t0=time.time(), sent=0, defaults=0, choices=0, asked=0, injected=None, errors=0, battles_done=0, msgs=0)
    client = poke_env.ps_client.PSClient
    real_send, patched_handle = client.send_message, client._handle_message

    async def send_message(self, message, room="", message_2=None):
        st["sent"] += 1
        if message == "/choose default":
            st["defaults"] += 1
        elif message.startswith("/choose"):
            st["choices"] += 1
        return await real_send(self, message, room, message_2)

    async def counted(self, message):
        st["msgs"] += 1
        if "|error|[Invalid choice]" in message:
            st["errors"] += 1
        return await patched_handle(self, message)

    client.send_message, client._handle_message = send_message, counted

    class Refused:
        """An order the server refuses: there is no move 9."""
        message = "/choose move 9"

    class Scripted(mod.SyncBridgePlayer):
        inject = False
        think = 0.3

        async def choose_move(self, battle):
            st["asked"] += 1
            self._battle_events.pop(battle.battle_tag, None)
            await asyncio.sleep(self.think)
            order = self.choose_random_move(battle)
            if (self.inject and st["injected"] is None and time.time() - st["t0"] > args.after and not battle.finished
                    and not battle.force_switch and battle.turn >= 3 and order.message.startswith("/choose move")):
                st["injected"] = dict(at=round(time.time() - st["t0"], 2), turn=battle.turn, room=battle.battle_tag)
                return Refused()
            return order

        def _battle_finished_callback(self, battle):
            st["battles_done"] += 1
            super()._battle_finished_callback(battle)

    async def make_semaphore():
        return asyncio.Semaphore(64)

    ep_sem = asyncio.run_coroutine_threadsafe(make_semaphore(), POKE_LOOP).result(10)
    events, futures, tag = queue.Queue(), {}, secrets.token_hex(3)

    def player(side):
        return Scripted(cfg=cfg, event_queue=events, action_futures=futures, ep_sem=ep_sem,
                        account_configuration=AccountConfiguration(f"s{tag}{side}", None),
                        server_configuration=mod.make_server_conf("127.0.0.1", args.port), max_concurrent_battles=1,
                        battle_format="gen9randombattle", start_listening=True, log_level=40)

    a, b = player("a"), player("b")
    a.inject, a.think, b.think = True, args.think_a, args.think_b

    async def start():
        for p in (a, b):
            await mod.wait_for_login(p)
        for p in (a, b):
            await mod.join_lobby(p)
        await mod.safe_send(a, f"/rlautospawn {a.username}, {b.username}, gen9randombattle, 1", room="lobby")

    asyncio.run_coroutine_threadsafe(start(), POKE_LOOP).result(60)

    def where(tasks, limit=4000):
        """The innermost coroutine each handler task is suspended in ("acquire" is the room's lock)."""
        counts: collections.Counter = collections.Counter()
        for task in list(tasks)[:limit]:
            name = "?"
            try:
                coro = task.get_coro()
                while coro is not None:
                    code = getattr(coro, "cr_code", None) or getattr(coro, "gi_code", None)
                    name = getattr(code, "co_name", type(coro).__name__)
                    inner = getattr(coro, "cr_await", None)
                    if inner is None:
                        inner = getattr(coro, "gi_yieldfrom", None)
                    if inner is None or not (hasattr(inner, "cr_code") or hasattr(inner, "gi_code")):
                        break
                    coro = inner
            except Exception:
                pass
            counts[name] += 1
        return dict(counts.most_common(5))

    proc = psutil.Process()
    last = dict(st)
    while True:
        time.sleep(1.0)
        try:
            while True:
                events.get_nowait()                  # nobody reads the worker's events here
        except queue.Empty:
            pass
        info = proc.memory_info()
        private = getattr(info, "private", info.rss) / 1e6
        tasks_a = set(a.ps_client._active_tasks)
        row = dict(up=round(time.time() - st["t0"], 1), private_mb=round(private, 1), rss_mb=round(info.rss / 1e6, 1),
                   cpu_s=round(sum(proc.cpu_times()[:2]), 1), tasks_a=len(tasks_a), tasks_b=len(b.ps_client._active_tasks),
                   where_a=where(tasks_a), battles_done=st["battles_done"],
                   turn_a=max([bt.turn for bt in list(a._battles.values())] or [0]),
                   choices_refused=getattr(a, "choices_refused", None), rooms_closed=getattr(a, "rooms_closed", None),
                   repair=dict(getattr(mod, "CHOICE_REPAIR", {})), injected=st["injected"])
        for key in ("sent", "defaults", "choices", "asked", "errors", "msgs"):
            row[key] = st[key]
            row["d_" + key] = st[key] - last[key]
        last = dict(st)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        inj, up = st["injected"], time.time() - st["t0"]
        if private > args.cap_mb or (inj and up - inj["at"] > args.watch) or up > args.after + args.watch + 240:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"end": "memory cap" if private > args.cap_mb else "time", "up": round(up, 1)}) + "\n")
            os._exit(0)                              # poke-env's loop thread would keep the process alive


if __name__ == "__main__":
    sys.exit(main())
