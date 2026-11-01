"""
Memory diagnostics for a rollout worker. Off unless the environment names a directory:

    PSPPO_MEMWATCH=<dir>            every worker writes <dir>/worker_<port>_<pid>.jsonl, one line
                                    every PSPPO_MEMWATCH_SECONDS (30): its memory, the size of
                                    every container that holds something per battle, the count
                                    of live objects by type, and the episodes it has handed over
    PSPPO_MEMWATCH_TRACE=<port>|all that worker also runs tracemalloc and writes its biggest
                                    allocation sites every fourth sample (slow: one worker only)

Nothing here changes what the worker does. bench/MEMORY.md says what was found with it, and
bench/memsample.py samples the same run from outside (every process, servers included).
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import sys
import threading
import time
from collections import Counter
from typing import Any, Dict, Optional


def process_memory() -> Dict[str, int]:
    """Bytes this process holds. "private" is what only it holds (the figure to watch for a leak)."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        out = {"rss": int(info.rss)}
        private = getattr(info, "private", None)
        if private is not None:                     # Windows: the commit charge
            out["private"] = int(private)
        else:
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("RssAnon:"):
                            out["private"] = int(line.split()[1]) * 1024
            except OSError:
                pass
        return out
    except Exception:
        return {}


def _players(worker: Any) -> list:
    env = getattr(worker, "vec_env", None)
    return [p for pair in list(getattr(env, "active_pairs", [])) for p in pair]


def container_sizes(worker: Any) -> Dict[str, int]:
    """The length of everything in a rollout worker that is keyed by a battle, summed over its players."""
    out: Dict[str, int] = {}
    players = _players(worker)

    def total(name: str, fn) -> None:
        n = 0
        for p in players:
            try:
                n += int(fn(p))
            except Exception:
                out[name] = -1
                return
        out[name] = n

    total("battles", lambda p: len(p._battles))
    total("battle_starts", lambda p: len(p._battle_starts))
    total("last_act_time", lambda p: len(p._last_act_time))
    total("episode_slot_state", lambda p: len(p._episode_slot_state))
    total("battle_events_rooms", lambda p: len(p._battle_events))
    total("battle_events_lines", lambda p: sum(len(v) for v in list(p._battle_events.values())))
    total("finished_rooms", lambda p: len(p._finished_rooms))
    total("battle_locks", lambda p: len(p.ps_client._battle_locks))
    total("client_tasks", lambda p: len(p.ps_client._active_tasks))
    total("battle_count_queue", lambda p: p._battle_count_queue.qsize())
    total("battle_semaphore", lambda p: p._battle_semaphore._value)
    total("start_waiters", lambda p: len(getattr(p._battle_start_condition, "_waiters", ())))
    total("ws_recv_queue", lambda p: len(getattr(getattr(getattr(p.ps_client, "websocket", None), "recv_messages", None), "frames", ()) or ()))
    total("rooms_closed", lambda p: getattr(p, "rooms_closed", 0))
    total("choices_refused", lambda p: getattr(p, "choices_refused", 0))
    try:
        # "/choose default" sent after an invalid choice, and withheld (worker.CHOICE_REPAIR).
        out.update(getattr(sys.modules.get(type(worker).__module__), "CHOICE_REPAIR", {}))
    except Exception:
        pass
    env = getattr(worker, "vec_env", None)
    try:
        out["traj"] = len(worker._traj)
        out["traj_steps"] = sum(len(t["act"]) for t in list(worker._traj.values()))
    except Exception:
        out["traj"] = -1
    try:
        out["event_queue"] = env.event_queue.qsize()
        out["action_futures"] = len(env.action_futures)
        out["learner_queue"] = worker.learner_client.q.qsize()
    except Exception:
        pass
    try:
        out["asyncio_tasks"] = len(asyncio.all_tasks(env.async_loop))
    except Exception:
        out["asyncio_tasks"] = -1
    try:
        import utils
        out["norm_cache"] = len(utils._NORM_CACHE)
        out["unknown_entities"] = len(utils.UNKNOWN_ENTITIES)
    except Exception:
        pass
    out["loggers"] = len(logging.root.manager.loggerDict)
    out["threads"] = threading.active_count()
    return out


def type_counts(top: int = 45) -> Dict[str, int]:
    """Live objects the garbage collector tracks, by type. (Strings, numbers and numpy arrays are not in it.)"""
    counts: Counter = Counter()
    for obj in gc.get_objects():
        counts[type(obj).__name__] += 1
    return dict(counts.most_common(top))


class _Watch:
    def __init__(self, worker: Any, port: int, directory: str):
        self.worker = worker
        self.port = port
        self.path = os.path.join(directory, f"worker_{port}_{os.getpid()}.jsonl")
        self.period = float(os.environ.get("PSPPO_MEMWATCH_SECONDS", "30"))
        trace = os.environ.get("PSPPO_MEMWATCH_TRACE", "")
        self.trace = trace == "all" or trace == str(port)
        self.episodes = 0
        self.steps = 0
        self.started = time.time()
        self._count_episodes()
        if self.trace:
            import tracemalloc
            tracemalloc.start(8)
        threading.Thread(target=self._loop, daemon=True, name="memwatch").start()

    def _count_episodes(self) -> None:
        client = self.worker.learner_client
        submit = client.submit_episode

        def counted(obs, act, *rest):
            self.episodes += 1
            self.steps += int(len(act))
            return submit(obs, act, *rest)

        client.submit_episode = counted

    def _trace_top(self, n: int = 25) -> list:
        import tracemalloc
        snap = tracemalloc.take_snapshot()
        rows = []
        for stat in snap.statistics("traceback")[:n]:
            frames = [f"{os.path.basename(fr.filename)}:{fr.lineno}" for fr in stat.traceback]
            rows.append({"kb": stat.size // 1024, "n": stat.count, "at": frames[-6:]})
        return rows

    def _loop(self) -> None:
        i = 0
        while True:
            time.sleep(self.period)
            i += 1
            try:
                row: Dict[str, Any] = {"t": round(time.time(), 1), "up": round(time.time() - self.started, 1),
                                       "pid": os.getpid(), "port": self.port,
                                       "episodes": self.episodes, "steps": self.steps}
                row.update(process_memory())
                row["sizes"] = container_sizes(self.worker)
                row["types"] = type_counts()
                if self.trace and i % 4 == 0:
                    import tracemalloc
                    row["traced_kb"] = tracemalloc.get_traced_memory()[0] // 1024
                    row["trace"] = self._trace_top()
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
            except Exception as e:  # diagnostics must never stop a worker
                try:
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"t": time.time(), "error": repr(e)}) + "\n")
                except Exception:
                    pass


def start_if_asked(worker: Any, port: int) -> Optional[_Watch]:
    directory = os.environ.get("PSPPO_MEMWATCH", "")
    if not directory:
        return None
    try:
        os.makedirs(directory, exist_ok=True)
        return _Watch(worker, port, directory)
    except Exception as e:
        print(f"memwatch could not start: {e!r}", file=sys.stderr, flush=True)
        return None
