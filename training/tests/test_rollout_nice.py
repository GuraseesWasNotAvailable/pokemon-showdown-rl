"""
train.py --rollout-nice N: the CPU priorities of a run (scheduling only).

    python -m pytest tests/test_rollout_nice.py -q -p no:cacheprovider -rs

With the flag, Ray starts its workers at nice 0 instead of Ray's 15 (worker_niceness), so the
learner and the inference actors run at 0, and every rollout worker raises its own threads to N
(worker.RolloutWorker.set_nice). The Showdown servers get N from run.json (servers.command under
`nice -n N`), which is config only.

Held here:
  * the flag: absent it is off; 0 to 19 are taken, anything else is refused before Ray starts;
  * raise_thread_nice / set_nice, in a child process of their own (the test process keeps its
    nice): every thread started before is raised to N, a thread started afterwards is at N, a second
    call changes nothing, nothing is ever lowered, a setpriority that fails is counted and logged and
    never raised, and a thread the first walk missed is raised by the next heartbeat;
  * the heartbeat: without set_nice it is the dict it always was, key for key; with it, it carries
    the worker's nice;
  * train.main with a stand-in Ray: without the flag, ray.init is called exactly as before
    (ray.init(ignore_reinit_error=True)) and no worker is asked for set_nice, nothing is asked of the
    learner, and the telemetry loop gets the arguments it always got; with the flag, ray.init carries
    worker_niceness 0, every worker gets set_nice(N) before its run(), and a Ray that is already
    running (where Ray refuses a _system_config) is joined as before;
  * NiceWatch: a worker whose heartbeat has no nice (a new process) is asked again, once a minute at
    most;
  * the processes a worker starts (the opponent seats' helpers: Foul Play's in a session of their
    own, PSBot's node engine), which with the flag start at Ray's 0 when the worker's constructor
    starts them: set_nice raises every thread of every process under the worker (children,
    grandchildren, a process another thread started), never lowers one, finds them through
    /proc/<pid>/task/<tid>/children or, without it, every /proc/<pid>/stat, and the heartbeat walks
    them again; with the real PSBot engine and Foul Play helpers where those are installed;
  * with a real local Ray (skipped where Ray cannot start): worker_niceness 0 starts an actor and the
    helper its constructor starts at 0, set_nice puts all its threads and the helper at N, and an
    actor killed by its own pid and started again by Ray is put back at N, its new helper with it,
    through its heartbeat;
  * run.json: the live config with servers.command under nice and --rollout-nice in the ppo phase's
    extra_train_args passes the supervisor's own load_config, its server command is the default one
    behind `nice -n 5`, and its trainer command passes train.py's parser.

Nothing here starts a server or a training run, and nothing touches the learner or the update.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "tests"), os.path.join(REPO, "cloud")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

LINUX = os.path.isdir("/proc/self/task") and hasattr(os, "setpriority")
needs_linux = pytest.mark.skipif(not LINUX, reason="per-thread nice values need Linux (/proc/self/task, setpriority)")


def _own_nice() -> int:
    return os.getpriority(os.PRIO_PROCESS, 0)


# ------------------------------------------------------------------------------------ the flag

def test_absent_the_flag_is_off_and_bad_values_are_refused():
    import train
    assert train.parse_args([]).rollout_nice is None
    assert train.rollout_nice_of(train.parse_args([])) is None
    assert train.rollout_nice_of(SimpleNamespace()) is None          # a Namespace built by hand, without it
    for n in (0, 5, 19):
        assert train.rollout_nice_of(train.parse_args(["--rollout-nice", str(n)])) == n
    for bad in ("-1", "20", "x"):
        with pytest.raises(SystemExit):
            train.parse_args(["--rollout-nice", bad])


# --------------------------------------------------------------- raise_thread_nice and set_nice

CHILD = textwrap.dedent(r'''
    import json, logging, os, queue, sys, threading
    sys.path.insert(0, sys.argv[1])
    os.chdir(sys.argv[1])
    import worker
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    def nices():
        out = {}
        for t in os.listdir("/proc/self/task"):
            try:
                out[int(t)] = os.getpriority(os.PRIO_PROCESS, int(t))
            except ProcessLookupError:
                pass
        return out

    base = os.getpriority(os.PRIO_PROCESS, 0)
    n = min(19, base + 3)
    stop = threading.Event()
    def park():
        stop.wait()
    early = [threading.Thread(target=park, daemon=True) for _ in range(3)]
    for t in early:
        t.start()
    res = {"base": base, "n": n, "before": sorted(set(nices().values()))}

    # A rollout worker without its servers: the heartbeat reads only these.
    w = worker.RolloutWorker.__new__(worker.RolloutWorker)
    w.vec_env = type("V", (), {"active_pairs": [], "ep_sem": None})()
    w.learner_client = type("C", (), {"q": queue.Queue()})()
    w._traj = {}
    res["beat_off"] = w.heartbeat()

    res["first"] = w.set_nice(n)
    res["after_first"] = sorted(set(nices().values()))
    res["early_at_n"] = all(os.getpriority(os.PRIO_PROCESS, t.native_id) == n for t in early)
    late = threading.Thread(target=park, daemon=True)
    late.start()
    res["late_nice"] = os.getpriority(os.PRIO_PROCESS, late.native_id)
    res["second"] = w.set_nice(n)
    res["lower"] = worker.raise_thread_nice(n - 1) if n > 0 else {"raised": 0}
    res["after_lower"] = sorted(set(nices().values()))

    # A thread the walk did not see (started during it): the next heartbeat raises it.
    if n < 19:
        missed = threading.Thread(target=park, daemon=True)
        missed.start()
        w._rollout_nice = n + 1             # as if this thread had been started below the nice
        listdir = os.listdir
        def hide(path, _real=listdir, _tid=str(missed.native_id)):
            names = _real(path)
            return [x for x in names if x != _tid] if path == "/proc/self/task" else names
        os.listdir = hide
        hidden = w._nice_walk()
        os.listdir = listdir
        res["hidden_raised"] = hidden["raised"]
        res["missed_before_beat"] = os.getpriority(os.PRIO_PROCESS, missed.native_id)
        beat = w.heartbeat()
        res["beat_on"] = beat
        res["missed_after_beat"] = os.getpriority(os.PRIO_PROCESS, missed.native_id)
        top = n + 1
    else:
        top = n

    # A setpriority that fails: counted, logged, never raised.
    real = os.setpriority
    def refuse(*a):
        raise PermissionError(13, "refused for the test")
    os.setpriority = refuse
    res["failing"] = worker.raise_thread_nice(min(19, top + 1)) if top < 19 else None
    w._rollout_nice = min(19, top + 1)
    try:
        res["failing_beat"] = w.heartbeat().get("nice")
    except Exception as e:
        res["failing_beat"] = "RAISED " + repr(e)
    os.setpriority = real
    stop.set()
    print("RESULT " + json.dumps(res))
''')


@needs_linux
def test_set_nice_raises_every_thread_and_later_threads_inherit_it():
    if _own_nice() > 14:
        pytest.skip("this process runs at nice above 14: no room to raise a child by 3 and more")
    p = subprocess.run([sys.executable, "-c", CHILD, REPO], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    assert line, (p.stdout[-2000:], p.stderr[-2000:])
    r = json.loads(line[-1][len("RESULT "):])
    n = r["n"]
    assert r["before"] == [r["base"]]

    # Without set_nice the heartbeat is the dict it always was.
    assert sorted(r["beat_off"]) == ["active_battles_worker", "ep_sem_value", "learner_q_size", "traj_in_memory"]

    # Every thread (main, the three started earlier, Python's and the libraries' own) is at n.
    first = r["first"]
    assert first["nice"] == n and first["failed"] == 0 and first["threads"] >= 4 and first["raised"] == first["threads"]
    assert r["after_first"] == [n] and r["early_at_n"]
    # A thread started afterwards inherits it, a second call changes nothing, nothing is lowered.
    assert r["late_nice"] == n
    assert r["second"]["raised"] == 0 and r["second"]["failed"] == 0 and r["second"]["threads"] >= 5
    assert r["lower"]["raised"] == 0 and r["after_lower"] == [n]

    # A thread the walk missed is raised by the next heartbeat, which reports its nice.
    if "beat_on" in r:
        assert r["hidden_raised"] >= 1 and r["missed_before_beat"] == n
        assert r["missed_after_beat"] == n + 1
        beat = r["beat_on"]
        assert sorted(beat) == ["active_battles_worker", "ep_sem_value", "learner_q_size", "nice", "traj_in_memory"]
        assert beat["nice"]["nice"] == n + 1 and beat["nice"]["raised_later"] >= 1 and beat["nice"]["failed"] == 0

    # A failing setpriority: counted, its text kept (three at most), never raised.
    if r["failing"] is not None:
        f = r["failing"]
        assert f["raised"] == 0 and f["failed"] == f["threads"] >= 4 and 1 <= len(f["errors"]) <= 3
        assert "refused for the test" in f["errors"][0]
    assert isinstance(r["failing_beat"], dict) and r["failing_beat"]["failed"] >= 1
    assert "not raised" in p.stderr and "refused for the test" in p.stderr


def test_raise_thread_nice_never_raises_without_proc(monkeypatch):
    import worker

    def no_proc(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(worker.os, "listdir", no_proc)
    out = worker.raise_thread_nice(5)
    assert out["threads"] == 0 and out["raised"] == 0 and out["failed"] == 1 and "cannot list" in out["errors"][0]


def test_raise_children_nice_never_raises_without_proc(monkeypatch):
    import worker

    def no_proc(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(worker.os, "listdir", no_proc)
    monkeypatch.setattr(worker, "_CHILDREN_FILE", False)
    out = worker.raise_children_nice(5)
    assert out["children"] == 0 and out["raised"] == 0 and out["failed"] == 1
    assert "cannot list the child processes" in out["errors"][0]
    # a child process that has ended: nothing counted, no failure
    gone = worker.raise_thread_nice(5, pid=999999999)
    assert gone.get("gone") is True and gone["threads"] == 0 and gone["failed"] == 0


def test_ppid_of_stat_reads_past_a_command_name_with_spaces_and_parentheses():
    import worker
    assert worker._ppid_of_stat("4242 (sleep) S 17 4242 17 0 -1 4194304") == 17
    assert worker._ppid_of_stat("4243 (a) b (c) d) R 99 4243 1 0") == 99
    assert worker._ppid_of_stat("4244 (ray::Opponent Seats) S 2138158 1 1 0") == 2138158


# A rollout worker that has started helper processes before its set_nice, as SeatsVectorWrapper's
# constructor does (Foul Play's helpers in a session of their own, PSBot's node engine): with
# --rollout-nice they start at Ray's 0, the learner's nice, and set_nice has to raise them too.
CHILD_PROCS = textwrap.dedent(r'''
    import json, os, queue, signal, subprocess, sys, threading, time
    sys.path.insert(0, sys.argv[1])
    os.chdir(sys.argv[1])
    import worker

    def nices(pid):
        out = set()
        try:
            names = os.listdir(f"/proc/{pid}/task")
        except FileNotFoundError:
            return None
        for t in names:
            try:
                out.add(os.getpriority(os.PRIO_PROCESS, int(t)))
            except ProcessLookupError:
                pass
        return sorted(out)

    def alive(pid):
        try:
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(") ", 1)[1][:1] != "Z"
        except FileNotFoundError:
            return False

    def tree(root):
        """Every process under root, read here from each /proc/<pid>/stat (not with the code under test)."""
        kids = {}
        for name in os.listdir("/proc"):
            if name.isdigit():
                try:
                    with open(f"/proc/{name}/stat") as f:
                        kids.setdefault(int(f.read().rsplit(") ", 1)[1].split()[1]), []).append(int(name))
                except (OSError, IndexError, ValueError):
                    pass
        out, todo = [], [root]
        while todo:
            for k in kids.get(todo.pop(0), []):
                out.append(k)
                todo.append(k)
        return sorted(out)

    HELPER = ("import subprocess, sys, threading\n"
              "g = subprocess.Popen(['sleep', '600'])\n"
              "ev = threading.Event()\n"
              "ts = [threading.Thread(target=ev.wait, daemon=True) for _ in range(3)]\n"
              "[t.start() for t in ts]\n"
              "print(g.pid, flush=True)\n"
              "sys.stdin.read()\n"
              "g.kill(); g.wait()\n")

    base = os.getpriority(os.PRIO_PROCESS, 0)
    n = min(18, base + 3)
    res = {"base": base, "n": n}
    popens, pids = [], []
    threaded = None
    try:
        own_session = subprocess.Popen(["sleep", "600"], start_new_session=True)       # as Foul Play's helpers
        popens.append(own_session)
        threaded = subprocess.Popen([sys.executable, "-c", HELPER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    text=True)                                         # an engine with threads and a child
        popens.append(threaded)
        grand = int(threaded.stdout.readline())
        pids.append(grand)
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("p", subprocess.Popen(["sleep", "600"])))
        t.start()
        t.join()                                                                       # started by another thread
        popens.append(box["p"])
        high = subprocess.Popen(["nice", "-n", "19", "sleep", "600"])                  # already above n: never lowered
        popens.append(high)
        kids = [own_session.pid, threaded.pid, grand, box["p"].pid, high.pid]
        time.sleep(0.5)
        res["kids"] = sorted(kids)
        res["tree"] = tree(os.getpid())
        res["high"] = high.pid
        res["before"] = {str(p): nices(p) for p in kids}

        w = worker.RolloutWorker.__new__(worker.RolloutWorker)
        w.vec_env = type("V", (), {"active_pairs": [], "ep_sem": None})()
        w.learner_client = type("C", (), {"q": queue.Queue()})()
        w._traj = {}
        res["first"] = w.set_nice(n)
        res["after"] = {str(p): nices(p) for p in kids}
        find = getattr(worker, "descendant_pids", None)
        if find is not None:
            res["by_file"] = sorted(find())
            worker._CHILDREN_FILE = False
            res["by_stat"] = sorted(find())
            worker._CHILDREN_FILE = None
        later = subprocess.Popen(["sleep", "600"])
        popens.append(later)
        res["later"] = nices(later.pid)
        res["second"] = w.set_nice(n)
        # The heartbeat walks the processes again, here through the /proc/<pid>/stat fallback: as if
        # they had all started below the nice, it is put one higher.
        worker._CHILDREN_FILE = False
        w._rollout_nice = n + 1
        res["beat"] = w.heartbeat()["nice"]
        res["after_beat"] = {str(p): nices(p) for p in kids + [later.pid]}
        worker._CHILDREN_FILE = None
    finally:
        if threaded is not None:
            try:
                threaded.stdin.close()
            except Exception:
                pass
        for p in popens:
            try:
                p.kill()
            except Exception:
                pass
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in popens:
            try:
                p.wait(timeout=30)
            except Exception:
                pass
        time.sleep(0.3)
        res["left"] = [p for p in pids + [x.pid for x in popens] if alive(p)]
    print("RESULT " + json.dumps(res))
''')


@needs_linux
def test_set_nice_raises_the_processes_the_worker_started_before_it():
    if _own_nice() > 14:
        pytest.skip("this process runs at nice above 14: no room to raise a child by 4")
    p = subprocess.run([sys.executable, "-c", CHILD_PROCS, REPO], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    assert line, (p.stdout[-2000:], p.stderr[-2000:])
    r = json.loads(line[-1][len("RESULT "):])
    n, base, high = r["n"], r["base"], str(r["high"])
    kids = [str(k) for k in r["kids"]]
    assert r["left"] == []                                           # every process the test started is gone
    assert r["tree"] == r["kids"]                                    # nothing else runs under the test process
    # Before set_nice the helpers are at the worker's nice (with the flag, Ray's 0): the defect.
    assert all(r["before"][k] == [base] for k in kids if k != high) and r["before"][high] == [19]
    # After it, every thread of each is at n (children, grandchild, other session, other thread); never lowered.
    assert all(r["after"][k] == [n] for k in kids if k != high) and r["after"][high] == [19], r["after"]
    # Both ways of finding them find exactly them.
    assert r["by_file"] == r["kids"] and r["by_stat"] == r["kids"]

    first = r["first"]
    assert first["children"] == 5 and first["failed"] == 0
    assert first["child_threads"] == 8 and first["child_raised"] == 7   # 1 + 4 + 1 + 1 raised; the nice-19 one left
    assert r["later"] == [n]                                         # started afterwards: inherits it
    assert r["second"]["child_raised"] == 0 and r["second"]["raised"] == 0 and r["second"]["children"] == 6

    beat = r["beat"]
    assert beat["children"] == 6 and beat["child_threads"] == 9 and beat["failed"] == 0
    assert beat["raised_later"] >= 8                                 # the 8 helper threads below n + 1, and the worker's own
    assert all(v == [n + 1] for k, v in r["after_beat"].items() if k != high) and r["after_beat"][high] == [19]


# The real opponent-seat helpers, started the way SeatsVectorWrapper's constructor starts them.
CHILD_SEATS = textwrap.dedent(r'''
    import json, os, queue, sys, time
    sys.path.insert(0, sys.argv[1])
    os.chdir(sys.argv[1])
    import worker, opponent_seats

    def nices(pid):
        out = set()
        try:
            names = os.listdir(f"/proc/{pid}/task")
        except FileNotFoundError:
            return None
        for t in names:
            try:
                out.add(os.getpriority(os.PRIO_PROCESS, int(t)))
            except ProcessLookupError:
                pass
        return sorted(out)

    def alive(pid):
        try:
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(") ", 1)[1][:1] != "Z"
        except FileNotFoundError:
            return False

    def tree(root):
        """Every process under root, read here from each /proc/<pid>/stat (not with the code under test)."""
        kids = {}
        for name in os.listdir("/proc"):
            if name.isdigit():
                try:
                    with open(f"/proc/{name}/stat") as f:
                        kids.setdefault(int(f.read().rsplit(") ", 1)[1].split()[1]), []).append(int(name))
                except (OSError, IndexError, ValueError):
                    pass
        out, todo = [], [root]
        while todo:
            for k in kids.get(todo.pop(0), []):
                out.append(k)
                todo.append(k)
        return sorted(out)

    base = os.getpriority(os.PRIO_PROCESS, 0)
    n = min(19, base + 5)
    res = {"base": base, "n": n, "kinds": []}
    bridge = None
    try:
        if os.environ.get("TEST_PSBOT") == "1":
            why = opponent_seats._psbot_usable()                       # as the constructor: the shared engine
            res["psbot_why"] = why
            if not why:
                res["kinds"].append("psbot")
        if os.environ.get("TEST_FOULPLAY") == "1":
            why = opponent_seats._foulplay_usable()
            res["foulplay_why"] = why
            if not why:
                bridge = opponent_seats._make_bridge(opponent_seats.OpponentSeatsConfig().foulplay, 2)
                bridge.start(wait=True)                                # the constructor does not wait; here they are up
                res["kinds"].append("foulplay")
        time.sleep(1.0)
        kids = tree(os.getpid())
        res["before"] = {str(p): nices(p) for p in kids}
        w = worker.RolloutWorker.__new__(worker.RolloutWorker)
        w.vec_env = type("V", (), {"active_pairs": [], "ep_sem": None})()
        w.learner_client = type("C", (), {"q": queue.Queue()})()
        w._traj = {}
        res["set_nice"] = w.set_nice(n)
        kids = tree(os.getpid())
        res["after"] = {str(p): nices(p) for p in kids}
        res["own"] = nices(os.getpid())
    finally:
        if bridge is not None:
            bridge.close()
        if "psbot" in res["kinds"]:
            import psbot_bridge
            psbot_bridge.shared_engine().close()
        time.sleep(1.0)
        res["left"] = [p for p in tree(os.getpid()) if alive(p)]
    print("RESULT " + json.dumps(res))
''')


@needs_linux
def test_set_nice_raises_the_real_psbot_engine_and_foul_play_helpers():
    """
    PSBot's engine needs node and its client cache (PSBOT_CACHE, or heuristic/cache in this tree;
    nothing is downloaded here); Foul Play needs its installation (FOULPLAY_DIR, default ~/foul-play,
    with its .venv). A part that is not there is skipped.
    """
    if _own_nice() > 14:
        pytest.skip("this process runs at nice above 14")
    env = dict(os.environ)
    cache = env.get("PSBOT_CACHE") or os.path.join(REPO, "heuristic", "cache")
    env["TEST_PSBOT"] = "1" if os.path.isfile(os.path.join(cache, "battle.js")) else "0"
    env["TEST_FOULPLAY"] = "1"
    p = subprocess.run([sys.executable, "-c", CHILD_SEATS, REPO], capture_output=True, text=True, timeout=600, env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    assert line, (p.stdout[-2000:], p.stderr[-2000:])
    r = json.loads(line[-1][len("RESULT "):])
    assert r["left"] == []                                           # the engine and the helpers are stopped
    if not r["kinds"]:
        pytest.skip(f"neither PSBot's engine nor Foul Play is usable here: {r.get('psbot_why')!r} {r.get('foulplay_why')!r}")
    n, base = r["n"], r["base"]
    assert r["before"] and all(v == [base] for v in r["before"].values() if v is not None)   # the defect
    assert r["own"] == [n]
    assert r["after"] and all(v == [n] for v in r["after"].values() if v is not None), r["after"]
    s = r["set_nice"]
    assert s["failed"] == 0 and s["children"] >= len(r["kinds"]) and s["child_raised"] == s["child_threads"] > 0
    print(f"real helpers {r['kinds']}: {len(r['after'])} processes, {s['child_threads']} threads raised from {base} to {n}")


# ------------------------------------------------------------------- train.main with a stand-in Ray

class _Ref:
    def __init__(self, value):
        self.value = value


class _Method:
    def __init__(self, actor, name):
        self.actor, self.name = actor, name

    def remote(self, *a, **k):
        self.actor.ray.calls.append((self.actor.kind, self.actor.idx, self.name, a, k))
        if self.name == "set_nice":
            return _Ref({"pid": 1000 + self.actor.idx, "nice": a[0], "threads": 30, "raised": 30, "failed": 0})
        if self.name == "__ray_call__":
            return _Ref((2000 + self.actor.idx, 0))
        if self.name == "heartbeat":
            return _Ref({"active_battles_worker": 0})
        return _Ref(None)


class _Actor:
    def __init__(self, ray, kind, idx):
        self.ray, self.kind, self.idx = ray, kind, idx

    def __getattr__(self, name):
        if name.startswith("__") and name != "__ray_call__":
            raise AttributeError(name)
        return _Method(self, name)


class _RemoteClass:
    def __init__(self, ray, cls):
        self.ray, self.kind = ray, getattr(cls, "__name__", str(cls))

    def options(self, **_k):
        return self

    def remote(self, *a, **k):
        actor = _Actor(self.ray, self.kind, len([c for c in self.ray.actors if c.kind == self.kind]))
        self.ray.actors.append(actor)
        self.ray.calls.append((self.kind, actor.idx, "__init__", a, k))
        return actor


class FakeRay:
    """The parts of ray that train.main uses, recording every call."""

    def __init__(self, refuse_system_config: bool = False):
        self.inits = []
        self.calls = []
        self.actors = []
        self.waits = 0
        self.refuse_system_config = refuse_system_config

    def init(self, **kwargs):
        self.inits.append(kwargs)
        if self.refuse_system_config and kwargs.get("_system_config"):
            raise ValueError("When connecting to an existing cluster, _system_config must not be provided.")

    def put(self, x):
        return _Ref(x)

    def cluster_resources(self):
        return {"GPU": 0.0, "CPU": 4.0}

    def remote(self, *args, **kwargs):
        if args and not kwargs:
            return _RemoteClass(self, args[0])
        return lambda cls: _RemoteClass(self, cls)

    def get(self, refs, timeout=None):
        return [r.value for r in refs] if isinstance(refs, list) else refs.value

    def wait(self, refs, num_returns=1, timeout=None):
        self.waits += 1
        return list(refs), []

    def shutdown(self):
        self.calls.append(("ray", 0, "shutdown", (), {}))

    def is_initialized(self):
        return bool(self.inits)


def _run_main(monkeypatch, argv, fake):
    import train
    seen = {}

    async def telemetry(*a, **k):
        seen["args"], seen["kwargs"] = a, k

    monkeypatch.setattr(train, "ray", fake)
    monkeypatch.setattr(train, "WeightStore", _RemoteClass(fake, type("WeightStore", (), {})))
    monkeypatch.setattr(train, "_telemetry_loop", telemetry)
    monkeypatch.setattr(train, "time", SimpleNamespace(sleep=lambda s: None, time=time.time))
    args = train.parse_args(["--ports", "11460-11461", "--battles", "64", "--rooms-per-pair", "16"] + argv)
    asyncio.run(train.main(args))
    return seen


def test_without_the_flag_main_is_as_before(monkeypatch):
    fake = FakeRay()
    seen = _run_main(monkeypatch, [], fake)
    assert fake.inits == [{"ignore_reinit_error": True}]            # ray.init(ignore_reinit_error=True), as 4cfd496
    names = [c[2] for c in fake.calls]
    assert "set_nice" not in names and "__ray_call__" not in names and fake.waits == 0
    assert names.count("run") == 2
    assert set(seen["kwargs"]) == set()                              # _telemetry_loop(args, started, infer, learner, workers, run_refs)
    assert len(seen["args"]) == 6


def test_with_the_flag_ray_starts_workers_at_zero_and_every_worker_gets_its_nice_first(monkeypatch, caplog):
    fake = FakeRay()
    with caplog.at_level("INFO", logger="train"):
        seen = _run_main(monkeypatch, ["--rollout-nice", "5"], fake)
    assert fake.inits == [{"ignore_reinit_error": True, "_system_config": {"worker_niceness": 0}}]
    workers = [c for c in fake.calls if c[0] == "RolloutWorker"]
    order = [(c[1], c[2]) for c in workers if c[2] in ("set_nice", "run")]
    assert sorted(order) == [(0, "run"), (0, "set_nice"), (1, "run"), (1, "set_nice")]
    # every set_nice before any run (the rollout thread starts at the nice), each asked once, with 5
    assert [x[1] for x in order] == ["set_nice", "set_nice", "run", "run"]
    assert all(c[3] == (5,) for c in workers if c[2] == "set_nice")
    asked = [c for c in fake.calls if c[2] == "__ray_call__"]
    assert sorted(c[0] for c in asked) == ["InferenceActor", "LearnerActor"]
    nice = seen["kwargs"].get("nice")
    assert nice is not None and nice.n == 5 and sorted(nice.answers) == [0, 1] and not nice.pending
    text = caplog.text
    assert "rollout nice 5: 2 of 2 workers at nice 5 (60 threads, 60 raised, 0 could not be)" in text
    assert "learner pid 2000 nice 0; inference pids [2000] nice [0]" in text
    assert "CPU priorities: Ray's workers start at nice 0" in text


def test_with_the_flag_a_running_cluster_is_joined_as_before(monkeypatch, caplog):
    fake = FakeRay(refuse_system_config=True)
    with caplog.at_level("INFO", logger="train"):
        seen = _run_main(monkeypatch, ["--rollout-nice", "5"], fake)
    assert fake.inits == [{"ignore_reinit_error": True, "_system_config": {"worker_niceness": 0}},
                          {"ignore_reinit_error": True}]
    assert "already running" in caplog.text and "nice of the running cluster" in caplog.text
    assert [c[2] for c in fake.calls if c[0] == "RolloutWorker"].count("set_nice") == 2
    assert seen["kwargs"]["nice"].n == 5


def test_another_ray_error_is_not_swallowed(monkeypatch):
    import train

    class Broken(FakeRay):
        def init(self, **kwargs):
            self.inits.append(kwargs)
            raise ValueError("something else")

    fake = Broken()
    monkeypatch.setattr(train, "ray", fake)
    with pytest.raises(ValueError, match="something else"):
        train.init_ray_with_priorities()
    assert len(fake.inits) == 1


def test_nicewatch_asks_a_worker_without_its_nice_again_once_a_minute(monkeypatch, caplog):
    import train
    fake = FakeRay()
    monkeypatch.setattr(train, "ray", fake)
    clock = {"t": 1000.0}
    monkeypatch.setattr(train, "time", SimpleNamespace(sleep=lambda s: None, time=lambda: clock["t"]))
    workers = [_Actor(fake, "RolloutWorker", i) for i in range(3)]
    w = train.NiceWatch(5, workers, [11460, 11461, 11462])
    with caplog.at_level("INFO", logger="train"):
        w.start()
        assert sorted(w.answers) == [0, 1, 2] and not w.pending
        beats = [{"nice": {"nice": 5}}, {"active_battles_worker": 3}, {"nice": {"nice": 5}}]
        clock["t"] += 5
        w.after_beats(beats)                         # within a minute of the first ask: not again yet
        assert w.asked_again == 0
        clock["t"] += 60
        w.after_beats(beats)                         # worker 1 is a new process: asked again
        assert w.asked_again == 1
        clock["t"] += 5
        w.after_beats(beats)                         # its answer is collected and logged
        clock["t"] += 5
        w.after_beats(beats)                         # not again within a minute
        assert w.asked_again == 1
        w.after_beats([{"nice": {"nice": 5}}] * 3)   # every worker has it: nothing asked
        assert w.asked_again == 1
    sets = [c for c in fake.calls if c[2] == "set_nice"]
    assert [c[1] for c in sets] == [0, 1, 2, 1]
    assert "port(s) [11461] answer without their nice" in caplog.text
    assert "rollout nice 5: 1 of 1 workers at nice 5" in caplog.text


def test_nicewatch_never_raises_into_main_or_the_telemetry_loop(monkeypatch, caplog):
    """A Ray that fails under it: a warning, never an exception (the supervisor counts telemetry errors)."""
    import train

    class Failing(FakeRay):
        def wait(self, refs, num_returns=1, timeout=None):
            raise RuntimeError("ray.wait broke")

    fake = Failing()
    monkeypatch.setattr(train, "ray", fake)
    workers = [_Actor(fake, "RolloutWorker", i) for i in range(2)]
    w = train.NiceWatch(5, workers, [11460, 11461])
    with caplog.at_level("INFO", logger="train"):
        w.start()
        w.ASK_AGAIN_S = 0.0
        w.after_beats([{"active_battles_worker": 1}, {"active_battles_worker": 1}])
    assert "could not be asked for" in caplog.text and "could not be checked this time" in caplog.text
    assert "Telemetry loop encountered an error" not in caplog.text


# -------------------------------------------------------------------------------- a real local Ray

class _NiceProbe:
    """
    A stand-in rollout worker with RolloutWorker's set_nice and heartbeat, threads of its own, and a
    helper process started in its constructor in a session of its own, as SeatsVectorWrapper starts
    Foul Play's helpers (bridge.start(wait=False)) before set_nice can be called.
    """

    def __init__(self):
        import queue
        import threading
        import worker
        self._w = worker.RolloutWorker.__new__(worker.RolloutWorker)
        self._w.vec_env = SimpleNamespace(active_pairs=[], ep_sem=None)
        self._w.learner_client = SimpleNamespace(q=queue.Queue())
        self._w._traj = {}
        self._stop = threading.Event()
        for _ in range(3):
            threading.Thread(target=self._stop.wait, daemon=True).start()
        self._helper = subprocess.Popen(["sleep", "600"], start_new_session=True)

    def helper(self):
        """The helper's pid and the nice of its threads."""
        pid = self._helper.pid
        return pid, sorted({os.getpriority(os.PRIO_PROCESS, int(t)) for t in os.listdir(f"/proc/{pid}/task")})

    def stop_helper(self):
        self._helper.kill()
        self._helper.wait(timeout=30)
        return True

    def set_nice(self, n):
        return self._w.set_nice(n)

    def heartbeat(self):
        return self._w.heartbeat()

    def nices(self):
        out = set()
        for t in os.listdir("/proc/self/task"):
            try:
                out.add(os.getpriority(os.PRIO_PROCESS, int(t)))
            except ProcessLookupError:
                pass
        return os.getpid(), sorted(out)

    def start_thread(self):
        import threading
        t = threading.Thread(target=self._stop.wait, daemon=True)
        t.start()
        return os.getpriority(os.PRIO_PROCESS, t.native_id)


@needs_linux
def test_real_ray_worker_niceness_zero_set_nice_and_a_killed_worker_comes_back_at_its_nice(caplog):
    ray = pytest.importorskip("ray")
    import train
    if ray.is_initialized():
        ray.shutdown()
    if _own_nice() != 0:
        pytest.skip("this test process does not run at nice 0")
    os.environ["PYTHONPATH"] = os.pathsep.join([REPO, os.path.join(REPO, "tests")] +
                                               ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    try:
        ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False, log_to_driver=False,
                 _system_config={"worker_niceness": train.RAY_WORKER_NICENESS})
    except Exception as e:      # noqa: BLE001
        pytest.skip(f"Ray cannot start here: {e!r}")
    helpers = []
    try:
        W = ray.remote(num_cpus=0)(_NiceProbe).options(max_restarts=-1, max_task_retries=0)
        w = W.remote()
        pid0, nices0 = ray.get(w.nices.remote(), timeout=120)
        assert nices0 == [0]                                        # Ray started the actor at 0, not 15
        h0, hn0 = ray.get(w.helper.remote(), timeout=60)
        helpers.append(h0)
        assert hn0 == [0]                                           # its constructor's helper too: the learner's nice
        watch = train.NiceWatch(5, [w], [11460])
        with caplog.at_level("INFO", logger="train"):
            watch.start()
        assert ray.get(w.nices.remote(), timeout=60) == (pid0, [5])
        assert ray.get(w.helper.remote(), timeout=60) == (h0, [5])  # the helper is raised with the worker
        assert watch.answers[0]["nice"] == 5 and watch.answers[0]["failed"] == 0
        assert watch.answers[0]["children"] == 1 and watch.answers[0]["child_raised"] == 1
        assert "and 1 processes they started (1 threads, 1 raised)" in caplog.text
        assert ray.get(w.start_thread.remote(), timeout=60) == 5   # a thread started later inherits it
        assert ray.get(w.heartbeat.remote(), timeout=60)["nice"]["nice"] == 5

        # Killed by its own pid: Ray starts it again, a new process at 0 whose heartbeat has no nice.
        os.kill(pid0, signal.SIGKILL)
        deadline = time.time() + 120
        while True:
            try:
                pid1, nices1 = ray.get(w.nices.remote(), timeout=60)
                if pid1 != pid0:
                    break
            except Exception:       # noqa: BLE001  (the actor is being started again)
                pass
            assert time.time() < deadline, "Ray did not start the actor again"
            time.sleep(1.0)
        assert nices1 == [0]
        h1, hn1 = ray.get(w.helper.remote(), timeout=60)
        helpers.append(h1)
        assert h1 != h0 and hn1 == [0]
        beat = ray.get(w.heartbeat.remote(), timeout=60)
        assert "nice" not in beat
        watch.ASK_AGAIN_S = 0.0
        with caplog.at_level("INFO", logger="train"):
            watch.after_beats([beat])
            deadline = time.time() + 60
            while watch.pending and time.time() < deadline:
                watch.after_beats([{"nice": {}}])
                time.sleep(0.5)
        assert watch.asked_again == 1 and not watch.pending
        assert ray.get(w.nices.remote(), timeout=60) == (pid1, [5])
        assert ray.get(w.helper.remote(), timeout=60) == (h1, [5])  # the new process's helper as well
        assert "answer without their nice" in caplog.text
        ray.get(w.stop_helper.remote(), timeout=60)
        ray.kill(w)
    finally:
        ray.shutdown()
        for h in helpers:                                           # the killed actor's helper outlives it: stopped by pid
            try:
                os.kill(h, signal.SIGKILL)
            except ProcessLookupError:
                pass


# -------------------------------------------------------------------------------- run.json

LIVE_RUN_JSON = {
    "repo_dir": "/home/ubuntu/rel/v2-humanaux",
    "python": "/home/ubuntu/venv/bin/python",
    "showdown_dir": "/home/ubuntu/pokemon-showdown",
    "servers": {"count": 36, "base_port": 10600, "heap_mb": 2048, "ready_timeout_seconds": 420},
    "trainer": {"stop_timeout_seconds": 420, "failure_stop_timeout_seconds": 120, "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}},
    "train_defaults": {"battles": 2304, "rooms_per_pair": 16, "minibatch": 768, "seq_len": 256, "grad_accum": 1,
                       "steps_per_update": 36864, "teacher": "psbot", "save_every": 50,
                       "extra_train_args": ["--obs-matchup", "--obs-movesets", "--infer-actors", "3", "--infer-sync-s", "1"]},
    "phases": {"imitation": {"skip": False, "min_decisions": 12000000, "max_decisions": 12000000},
               "warmup": {"skip": False, "min_decisions": 500000, "max_decisions": 1500000},
               "ppo": {"skip": False, "target_decisions": 600000000,
                       "extra_train_args": ["--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1", "--read-bonus", "0.3",
                                            "--read-anneal", "20000000:45000000", "--opponent-seats", "setupnet:0.15",
                                            "--boosted-row-weight", "2", "--lr-hold-steps", "479713",
                                            "--lr-total-steps", "1113566"]}},
}

NICE_SERVERS_COMMAND = ["nice", "-n", "5", "{python}", "start_servers.py", "--dir", "{showdown_dir}", "--ports", "{ports}",
                        "--heap-mb", "{heap_mb}", "--log-dir", "{server_log_dir}"]


def _supervisor():
    """
    The supervisor module: PSPPO_LIVE_PIPELINE names a pipeline.py to load (the live one, copied from
    box a), else this repo's pipeline.py.
    """
    path = os.environ.get("PSPPO_LIVE_PIPELINE")
    if not path:
        import pipeline
        return pipeline
    import importlib.util
    spec = importlib.util.spec_from_file_location("live_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["live_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_json_with_the_nice_server_command_passes_the_supervisor(tmp_path):
    """
    PSPPO_LIVE_PIPELINE=<live pipeline.py> PSPPO_LIVE_RUN_JSON=<live run.json>: the same check on
    the live files (LIVE_RUN_JSON above is a copy of the live config's server and trainer parts).
    """
    import copy
    import train
    pipeline = _supervisor()

    def write(cfg, name):
        p = tmp_path / name
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return str(p)

    live = LIVE_RUN_JSON
    if os.environ.get("PSPPO_LIVE_RUN_JSON"):
        with open(os.environ["PSPPO_LIVE_RUN_JSON"], encoding="utf-8") as f:
            live = json.load(f)
        assert "command" not in live["servers"] and "--rollout-nice" not in json.dumps(live)
    before = copy.deepcopy(live)
    after = copy.deepcopy(live)
    after["servers"]["command"] = list(NICE_SERVERS_COMMAND)
    after["phases"]["ppo"]["extra_train_args"] += ["--rollout-nice", "5"]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = pipeline.Paths(str(run_dir))
    cfg0 = pipeline.load_config(write(before, "before.json"))
    cfg1 = pipeline.load_config(write(after, "after.json"))
    s0 = pipeline.build_server_command(cfg0, paths, "ppo")
    s1 = pipeline.build_server_command(cfg1, paths, "ppo")
    assert s1 == ["nice", "-n", "5"] + s0                    # the same launcher behind nice (nice execs it: same pid)
    assert s0[:2] == ["/home/ubuntu/venv/bin/python", "start_servers.py"] and "10600-10635" in s0
    t0 = pipeline.build_trainer_command(cfg0, paths, "ppo")
    t1 = pipeline.build_trainer_command(cfg1, paths, "ppo")
    assert t1 == t0 + ["--rollout-nice", "5"]
    args = train.parse_args(t1[2:])                          # train.py takes the command line
    assert args.rollout_nice == 5
    assert train.build_config(args) == train.build_config(train.parse_args(t0[2:]))   # the run's config is the same
