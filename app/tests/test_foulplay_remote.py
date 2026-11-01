"""
Tests for the remote helpers of foulplay_bridge.py (RemoteTarget: the search on another machine,
over ssh; ladder.py --search-remote; cloud/remote_search.sh). Linux only: they stand in for the
other machine with this one.

    python -m pytest tests/test_foulplay_remote.py -q -p no:cacheprovider

What stands in for the real things:
  * a fake ssh (FAKE_SSH below), started by the real code path (RemoteTarget.ssh). It parses ssh's
    command line as ssh does (options, the host, the rest joined into ONE command line) and runs
    that command line under a stand-in for sshd: a process of its own (its own session, as sshd
    makes it) that runs `sh -c <command>` and relays the lines. On request it delays every line
    by FAKE_SSH_DELAY_MS each way (the network's round trip), fails to connect (exit 255, as ssh
    does), or goes quiet without closing (FAKE_SSH_HALF_OPEN_AFTER_S: the client stops relaying and
    leaves FAKE_SSH_ALIVE_S later with 255, as ServerAliveCountMax makes ssh do; the sshd stand-in
    keeps the session open and notices nothing, as sshd on a dead connection). When the client is
    killed the stand-in goes as sshd does when the connection closes: it closes the command's
    stdin and leaves, so the command loses its parent.
  * the fake helper of tests/test_foulplay_bridge.py, as the command run "there", for the
    bridge's side (the command line, the probes, the deadlines, restarts, kills by tag).
  * the REAL helper (bench/fp_bridge_server.py with Foul Play, ~/foul-play) for what only it does:
    leaving when its ssh session goes, when the connection goes quiet, and its decisions. Skipped
    where Foul Play is not installed.

Nothing here deletes anything. Temporary folders are made under $FOULPLAY_TEST_TMP (default: the
system temp folder) and left there. Every process is started by foulplay_bridge.py or by a test,
and stopped by PID or by its tag.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import foulplay_bridge as fb  # noqa: E402
from foulplay_bridge import FoulPlayBridge, FoulPlayBridgeError, RemoteTarget  # noqa: E402

_spec = importlib.util.spec_from_file_location("tfb_for_remote", os.path.join(REPO, "tests", "test_foulplay_bridge.py"))
tfb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfb)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the other machine is stood in for with this one: Linux processes")

LAUNCHER = os.path.join(REPO, "launchers", "laptop", "Look Play Remote.cmd")
REMOTE_SH = os.path.join(REPO, "cloud", "remote_search.sh")

FAKE_SSH = r'''
import json, os, queue, signal, subprocess, sys, threading, time

TAKES = set("oipFlecLRDJmbEIwWBOQS")


def parse(argv):
    """ssh's command line: options (some with a value), the host, then the command (joined)."""
    i, host, opts = 0, None, []
    while i < len(argv):
        a = argv[i]
        if a.startswith("-") and len(a) > 1:
            if a[1] in TAKES and len(a) == 2:
                opts.append((a, argv[i + 1]))
                i += 2
            else:
                opts.append((a, None))
                i += 1
            continue
        host = a
        i += 1
        break
    return opts, host, " ".join(argv[i:])


def pump(src, dst, delay, stop, on_end=None):
    """Lines from src to dst, each `delay` s after it was read, in order."""
    q = queue.Queue()

    def reader():
        try:
            for line in iter(src.readline, b""):
                q.put((time.monotonic() + delay, line))
        except Exception:
            pass
        q.put((time.monotonic() + delay, None))

    threading.Thread(target=reader, daemon=True).start()
    while True:
        due, line = q.get()
        wait = due - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        if line is None:
            break
        if stop.is_set():
            continue                      # a connection gone quiet: nothing gets through
        try:
            dst.write(line)
            dst.flush()
        except Exception:
            break
    if on_end is not None:
        on_end()


def client():
    command = os.environ.pop("FAKE_SSH_COMMAND")
    delay = float(os.environ.get("FAKE_SSH_DELAY_MS") or 0) / 1000.0
    half = float(os.environ.get("FAKE_SSH_HALF_OPEN_AFTER_S") or 0)
    alive = float(os.environ.get("FAKE_SSH_ALIVE_S") or 3)
    if os.environ.get("FAKE_SSH_FAIL"):
        sys.stderr.write("ssh: connect to host %s port 22: Connection timed out\n" % os.environ.get("FAKE_SSH_HOST"))
        sys.exit(255)
    env = dict(os.environ, FAKE_SSHD_COMMAND=command)
    server = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--sshd"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True)
    stop = threading.Event()
    out, err = sys.stdout.buffer, sys.stderr.buffer

    def close_server_stdin():
        try:
            server.stdin.close()
        except Exception:
            pass

    threading.Thread(target=pump, args=(sys.stdin.buffer, server.stdin, delay, stop, close_server_stdin), daemon=True).start()
    threading.Thread(target=pump, args=(server.stderr, err, 0.0, stop), daemon=True).start()
    done = threading.Event()
    threading.Thread(target=lambda: (pump(server.stdout, out, delay, stop), done.set()), daemon=True).start()
    began = time.monotonic()
    while not done.is_set():
        if half and time.monotonic() - began > half and not stop.is_set():
            stop.set()
        if stop.is_set() and time.monotonic() - began > half + alive:
            err.write(b"Timeout, server not responding.\n")
            err.flush()
            os._exit(255)
        time.sleep(0.05)
    sys.exit(server.wait())


def sshd():
    command = os.environ.pop("FAKE_SSHD_COMMAND")
    half = float(os.environ.get("FAKE_SSH_HALF_OPEN_AFTER_S") or 0)
    for key in [k for k in os.environ if k.startswith("FAKE_SSH")]:
        os.environ.pop(key)
    child = subprocess.Popen(["sh", "-c", command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                             start_new_session=True)
    began, client_pid = time.monotonic(), os.getppid()
    stop = threading.Event()

    def end_of_input():
        if not (half and time.monotonic() - began > half):
            try:
                child.stdin.close()       # the channel's end of input
            except Exception:
                pass

    threading.Thread(target=pump, args=(sys.stdin.buffer, child.stdin, 0.0, stop, end_of_input), daemon=True).start()
    threading.Thread(target=pump, args=(child.stdout, sys.stdout.buffer, 0.0, stop), daemon=True).start()
    while child.poll() is None:
        quiet = half and time.monotonic() - began > half
        if quiet:
            stop.set()                    # a dead connection that nobody here has noticed: the session stays
        elif os.getppid() != client_pid:
            os._exit(255)                 # the connection closed: sshd leaves, the command loses its parent
        time.sleep(0.05)
    time.sleep(0.2)
    sys.exit(child.returncode)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sshd"]:
        sshd()
    elif "FAKE_SSH_COMMAND" in os.environ:
        client()
    else:
        # As ssh was called: note the command line, then run again with the command off the command
        # line (it holds the helper's tag, and pkill must find the helper there, not this process).
        log = os.environ.get("FAKE_SSH_LOG")
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps(sys.argv[1:]) + "\n")
        opts, host, command = parse(sys.argv[1:])
        env = dict(os.environ, FAKE_SSH_COMMAND=command, FAKE_SSH_HOST=str(host))
        os.execve(sys.executable, [sys.executable, "-u", os.path.abspath(__file__)], env)
'''


def new_dir(prefix: str) -> str:
    return tfb.new_dir(prefix)


@pytest.fixture(scope="module")
def fake_ssh() -> str:
    path = os.path.join(new_dir("fpremote_ssh_"), "fake_ssh.py")
    tfb.write(path, FAKE_SSH)
    return path


@pytest.fixture(scope="module")
def fake_helper() -> str:
    path = os.path.join(new_dir("fpremote_fake_"), "fake_helper.py")
    tfb.write(path, tfb.FAKE_HELPER)
    return path


@pytest.fixture
def scratch():
    """A new folder under $FOULPLAY_TEST_TMP, left there (pytest's own tmp folders: it removes older ones by itself)."""
    import pathlib

    return pathlib.Path(new_dir("fpremote_case_"))


@pytest.fixture(autouse=True)
def clean_ssh_env(monkeypatch):
    for key in [k for k in os.environ if k.startswith("FAKE_SSH")]:
        monkeypatch.delenv(key)


def target(fake_ssh: str, **kw) -> RemoteTarget:
    kw.setdefault("code_dir", REPO)
    return RemoteTarget(host="tester@box", key="/keys/box.pem", ssh=[sys.executable, "-u", fake_ssh],
                        known_hosts="/keys/known_hosts", **kw)


def remote_bridge(fake_ssh: str, fake_helper: str, helpers: int = 1, env=None, tgt=None, **kw) -> FoulPlayBridge:
    kw.setdefault("timeout", 20.0)
    return FoulPlayBridge(helpers=helpers, command=[sys.executable, "-u", fake_helper], env=env,
                          remote=tgt or target(fake_ssh), **kw)


def pids_with(pattern: str):
    out = subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.PIPE, text=True).stdout
    return [int(w) for w in out.split() if w.isdigit()]


def helper_pid(bridge: FoulPlayBridge, index: int = 0) -> int:
    return int(bridge._helpers[index].info["pid"])


# ---------------------------------------------------------------------------------------------
# the target and its command line
# ---------------------------------------------------------------------------------------------

def test_the_helper_is_one_command_line_after_the_host_with_its_tag_and_settings(fake_ssh, fake_helper, monkeypatch):
    log = os.path.join(new_dir("fpremote_log_"), "argv.jsonl")
    monkeypatch.setenv("FAKE_SSH_LOG", log)
    bridge = remote_bridge(fake_ssh, fake_helper, env={"FAKE_ITERATIONS_PER_MS": "7"}).start()
    try:
        argv = json.loads(open(log).read().splitlines()[0])
        host = argv.index("tester@box")
        assert len(argv) == host + 2                                   # the command line there is ONE argument
        options = argv[:host]
        for need in ("-T", "BatchMode=yes", "ServerAliveInterval=5", "ServerAliveCountMax=3", "ConnectTimeout=15",
                     "StrictHostKeyChecking=accept-new", "UserKnownHostsFile=/keys/known_hosts", "IdentitiesOnly=yes"):
            assert need in options, need
        assert options[options.index("-i") + 1] == "/keys/box.pem" and "-t" not in options
        line = argv[-1]
        helper = bridge._helpers[0]
        assert line.startswith("exec env PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FAKE_ITERATIONS_PER_MS=7 ")
        assert line.endswith("--tag " + helper.tag) and helper.tag.startswith(fb._REMOTE_TOKEN)
        assert helper.info["argv"][-2:] == ["--tag", helper.tag]          # what the helper there was given
        assert bridge.where == "remote" and bridge.remote_host == "tester@box"
        assert pids_with(fb._remote_pattern(helper.tag)) == [helper_pid(bridge)]   # found there by its tag, and only it
    finally:
        bridge.close()


def test_a_remote_target_from_user_at_host_or_from_vm_sh(scratch, monkeypatch):
    t = fb.remote_target("ubuntu@3.4.5.6", key="k.pem")
    assert (t.host, t.key, t.code_dir) == ("ubuntu@3.4.5.6", "k.pem", "~/dev/ab-ladder")
    for bad in ("", "a b@c", "-oProxyCommand=x"):
        with pytest.raises(FoulPlayBridgeError):
            fb.remote_target(bad)
    vm = scratch / "vm.sh"
    # What `vm.sh ssh -G` prints: ssh's resolved settings for the box (never a secret, only file names).
    vm.write_text('[ "$1 $2" = "ssh -G" ] || exit 9\n'
                  'printf "user ubuntu\\nhostname 54.1.2.3\\nport 22\\nidentityfile /home/me/.ssh/box.pem\\n'
                  'identityfile /home/me/.ssh/other\\nuserknownhostsfile /home/me/.ssh/psppo_known_hosts /home/me/.ssh/k2\\n"\n')
    monkeypatch.setenv("FOULPLAY_BASH", shutil.which("bash"))
    t = fb.remote_target("vm", vm_sh=str(vm), code_dir="~/dev/x")
    assert (t.host, t.key, t.known_hosts, t.port, t.code_dir) == (
        "ubuntu@54.1.2.3", "/home/me/.ssh/box.pem", "/home/me/.ssh/psppo_known_hosts", None, "~/dev/x")
    vm.write_text('echo "no running psppo-train box for target b in us-east-1 (state: stopped)" >&2; exit 1\n')
    with pytest.raises(FoulPlayBridgeError, match="is it running"):
        fb.remote_target("vm", vm_sh=str(vm))


def test_words_keep_a_leading_tilde_for_the_shell_there():
    assert RemoteTarget.line(["exec", "~/foul-play/.venv/bin/python", "a b", "--tag", "fpr.1.ab."]) == \
        "exec ~/foul-play/.venv/bin/python 'a b' --tag fpr.1.ab."
    assert fb._remote_pattern("fpr.12.ab.3.") == "[-]-tag fpr[.]12[.]ab[.]3[.]"
    assert re.search(fb._remote_pattern("fpr.12.ab."), "python x.py --tag fpr.12.ab.3.0.1.")
    assert not re.search(fb._remote_pattern("fpr.12.ab."), "pkill -f [-]-tag fpr[.]12[.]ab[.]")   # never itself


# ---------------------------------------------------------------------------------------------
# is it there
# ---------------------------------------------------------------------------------------------

def test_locate_remote_checks_foul_play_and_that_the_helper_there_is_this_one(fake_ssh, scratch):
    checkout = tfb.fake_checkout()
    code = scratch / "code"
    (code / "bench").mkdir(parents=True)
    tgt = target(fake_ssh, code_dir=str(code), fp_dir=checkout, python=sys.executable)
    ok, why, _ = fb.locate_remote(tgt)
    assert not ok and "is not on tester@box" in why and "remote_search.sh" in why
    (code / "bench" / "fp_bridge_server.py").write_bytes(b"# another version\n")
    ok, why, _ = fb.locate_remote(tgt)
    assert not ok and "not this repository's" in why
    # The same file with Windows line ends (a laptop checkout) is the same helper.
    (code / "bench" / "fp_bridge_server.py").write_bytes(fb.SERVER_PY.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    ok, why, info = fb.locate_remote(tgt)
    assert ok, why
    assert info["remote"] == "1" and info["server"] == str(code / "bench" / "fp_bridge_server.py") and int(info["cpus"]) >= 1
    assert info["dir"] == checkout and info["python"] == sys.executable and "on tester@box" in why and "cores there" in why
    assert fb.available(remote=tgt)[0]
    ok, why, _ = fb.locate_remote(target(fake_ssh, code_dir=str(code), fp_dir=str(scratch / "nothing")))
    assert not ok and "not installed on tester@box" in why


def test_a_box_that_cannot_be_reached_is_told_and_its_helpers_are_left_down(fake_ssh, fake_helper, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_FAIL", "1")
    ok, why, _ = fb.locate_remote(target(fake_ssh, fp_dir="/x"))
    assert not ok and "ssh itself failed" in why and "Connection timed out" in why
    bridge = remote_bridge(fake_ssh, fake_helper, start_timeout=10.0, max_restarts=1, restart_window=60.0, cooldown=60.0)
    try:
        with pytest.raises(FoulPlayBridgeError, match="Connection timed out|did not start|did not answer"):
            bridge.start(timeout=10.0)
    finally:
        bridge.close()


# ---------------------------------------------------------------------------------------------
# the round trip, the clock, the deadlines
# ---------------------------------------------------------------------------------------------

def test_the_round_trip_is_measured_at_start_and_kept_up_to_date(fake_ssh, fake_helper, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_DELAY_MS", "100")                     # 100 ms each way: 200 ms there and back
    bridge = remote_bridge(fake_ssh, fake_helper, helpers=2, env={"FAKE_CLOCK_OFFSET": "1000"}).start()
    try:
        trip = bridge.rtt_start
        assert trip["probes"] == 2 * fb.REMOTE_START_PROBES and len(trip["helpers"]) == 2
        assert 195 <= trip["min_ms"] <= trip["rtt_ms"] <= 330, trip
        for helper in bridge._helpers:
            assert helper.clock_ok and abs(helper.clock_offset - 1000.0) < 0.06     # its clock reads 1000 s more
            assert abs(helper.return_s - (helper.rtt_s / 2 + fb.REMOTE_MARGIN_S)) < 1e-9
        before = len(bridge._helpers[0].samples)
        assert tfb.wait_until(lambda: len(bridge._helpers[0].samples) > before, timeout=3 * fb.REMOTE_PROBE_S)
        stats = bridge.stats()
        assert stats["where"] == "remote" and 195 <= stats["rtt_ms"] <= 330 and stats["rtt_start_ms"] == trip["rtt_ms"]
        assert all(195 <= h["rtt_ms"] <= 330 for h in stats["helpers"])
        assert 195 <= bridge.round_trip_ms() <= 330
    finally:
        bridge.close()


def test_a_deadline_is_given_on_the_helpers_clock_less_the_way_back(fake_ssh, fake_helper, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_DELAY_MS", "100")
    bridge = remote_bridge(fake_ssh, fake_helper, env={"FAKE_CLOCK_OFFSET": "1000"}).start()
    try:
        tfb.feed_request(bridge, "u|battle-1", rqid=3)
        helper = bridge._helpers[0]
        asked_at = fb._clock()
        advice = bridge.advice_sync("u|battle-1", search_ms=100, rqid=3, timeout=3.0)
        deadline = advice.raw["asked"]["deadline"]
        # On the helper's clock (ours + 1000 s), 3 s from the question, less half the round trip and the margin.
        expected = asked_at + 1000.0 + 3.0 - (helper.rtt_s / 2.0 + fb.REMOTE_MARGIN_S)
        assert abs(deadline - expected) < 0.06, (deadline, expected)
        assert advice.wall_ms >= 200                                    # it went there and back
    finally:
        bridge.close()


def test_an_answer_that_could_not_come_back_in_time_is_not_searched(fake_ssh, fake_helper, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_DELAY_MS", "300")                     # 600 ms there and back
    bridge = remote_bridge(fake_ssh, fake_helper).start()
    try:
        tfb.feed_request(bridge, "u|battle-1", rqid=1)
        # 100 ms asks about 0.27 s of search: in 0.8 s it cannot be there, searched and back.
        with pytest.raises(FoulPlayBridgeError) as err:
            bridge.advice_sync("u|battle-1", search_ms=100, rqid=1, timeout=0.8)
        assert "did not answer" in str(err.value) or err.value.expired
        assert tfb.wait_until(lambda: bridge.counters["expired"] >= 1, timeout=3.0)   # the helper did not search it
        assert bridge.advice_sync("u|battle-1", search_ms=100, rqid=1, timeout=2.0).choice    # with time for the trip it is
    finally:
        bridge.close()


def test_early_expire_counts_the_round_trip(fake_ssh, fake_helper, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_DELAY_MS", "250")
    bridge = remote_bridge(fake_ssh, fake_helper, early_expire=True).start()
    try:
        tfb.feed_request(bridge, "u|battle-slow", rqid=1)
        tfb.feed_request(bridge, "u|battle-2", rqid=1)
        first = bridge.advice_future("u|battle-slow", search_ms=100, rqid=1, ttl=5.0)      # FAKE_SLOW_S: 1 s in the helper
        bridge._decide_s[0] = 0.3                                       # what one decision takes there, as measured
        # One decision ahead (0.3 s) + 0.27 s of search fits 0.9 s here, not with the 0.5 s round trip on top.
        second = bridge.advice_future("u|battle-2", search_ms=100, rqid=1, ttl=0.9)
        with pytest.raises(FoulPlayBridgeError) as err:
            second.result(timeout=1.0)
        assert err.value.expired and "decisions wait" in str(err.value) and bridge.counters["expired_early"] == 1
        first.result(timeout=5.0)
    finally:
        bridge.close()


# ---------------------------------------------------------------------------------------------
# going away: a dropped ssh, a hung helper, close(), this process killed
# ---------------------------------------------------------------------------------------------

def test_a_dropped_ssh_is_replaced_and_its_battles_rebuilt(fake_ssh, fake_helper):
    bridge = remote_bridge(fake_ssh, fake_helper).start()
    try:
        tfb.feed_request(bridge, "u|battle-1", rqid=1)
        assert bridge.advice_sync("u|battle-1", rqid=1).raw["feeds"] == 1
        old_ssh, old_pid = bridge._helpers[0].proc.pid, helper_pid(bridge)
        tfb.kill_hard(old_ssh)                                          # the connection goes
        assert tfb.wait_until(lambda: bridge._helpers[0] is not None and bridge._helpers[0].generation == 2
                              and bridge._helpers[0].replies > 0, timeout=20)
        assert tfb.wait_until(lambda: not tfb.pid_alive(old_pid), timeout=10)   # the old one there left (end of input)
        assert helper_pid(bridge) != old_pid and bridge._generation[0] == 2
        tfb.feed_request(bridge, "u|battle-1", rqid=2)
        advice = bridge.advice_sync("u|battle-1", rqid=2)
        assert advice.raw["feeds"] == 2 and bridge.counters["rebuilt_battles"] == 1      # rebuilt there from the stored lines
    finally:
        bridge.close()


def test_a_hung_remote_helper_is_stopped_there_by_its_tag(fake_ssh, fake_helper):
    bridge = remote_bridge(fake_ssh, fake_helper, hang_timeout=2.0, timeout=30.0).start()
    try:
        tfb.feed_request(bridge, "u|battle-hang", rqid=1)
        hung = helper_pid(bridge)
        future = bridge.advice_future("u|battle-hang", rqid=1)         # sleeps an hour: reads nothing, answers nothing
        assert tfb.wait_until(lambda: bridge.counters["hangs"] == 1, timeout=15)
        # Its ssh is stopped here, and the helper there (deaf to its end of input) by its tag.
        assert tfb.wait_until(lambda: not tfb.pid_alive(hung), timeout=20)
        with pytest.raises(FoulPlayBridgeError):
            future.result(timeout=20)
    finally:
        bridge.close()


def test_close_leaves_nothing_running_there(fake_ssh, fake_helper):
    tgt = target(fake_ssh)
    bridge = remote_bridge(fake_ssh, fake_helper, helpers=3, tgt=tgt).start()
    pids = [helper_pid(bridge, i) for i in range(3)]
    tag = "%s%d." % (fb._REMOTE_TOKEN, bridge._serial)
    assert sorted(fb.remote_leftovers(tgt, tag)) == sorted(pids)
    began = time.monotonic()
    bridge.close()
    assert time.monotonic() - began < fb.REMOTE_GRACE_S + 10
    assert fb.remote_leftovers(tgt, tag) == [] and not any(tfb.pid_alive(p) for p in pids)


def test_helpers_there_stop_when_the_process_that_started_them_is_killed(fake_ssh, fake_helper):
    script = os.path.join(new_dir("fpremote_parent_"), "parent.py")
    tfb.write(script, "\n".join([
        "import json, sys, time",
        "sys.path.insert(0, %r)" % REPO,
        "import foulplay_bridge as fb",
        "t = fb.RemoteTarget(host='tester@box', ssh=[sys.executable, '-u', %r], code_dir=%r)" % (fake_ssh, REPO),
        "b = fb.FoulPlayBridge(helpers=2, remote=t, command=[sys.executable, '-u', %r]).start()" % fake_helper,
        "print(json.dumps([h.info['pid'] for h in b._helpers]), flush=True)",
        "time.sleep(600)",
    ]))
    parent = subprocess.Popen([sys.executable, "-u", script], stdout=subprocess.PIPE, text=True)
    pids = []
    try:
        pids = json.loads(parent.stdout.readline())
        assert len(pids) == 2 and all(tfb.pid_alive(p) for p in pids)
        tfb.kill_hard(parent.pid)                                      # no close(), no atexit: its pipes just close
        parent.wait(timeout=10)
        assert tfb.wait_until(lambda: not any(tfb.pid_alive(p) for p in pids), timeout=20)
    finally:
        if parent.poll() is None:
            parent.kill()
        for pid in pids:
            if tfb.pid_alive(pid):
                tfb.kill_hard(pid)


def test_the_echo_measures_the_networks_round_trip(fake_ssh, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_DELAY_MS", "60")
    out = fb.remote_echo_round_trip(target(fake_ssh), pings=5, gap_s=0.02)
    assert out["ok"] and out["pings"] == 5 and 118 <= out["min_ms"] <= out["rtt_ms"] <= 250, out
    monkeypatch.setenv("FAKE_SSH_FAIL", "1")
    out = fb.remote_echo_round_trip(target(fake_ssh), pings=3)
    assert not out["ok"] and "Connection timed out" in out["error"]


# ---------------------------------------------------------------------------------------------
# the real helper there
# ---------------------------------------------------------------------------------------------

def real_foul_play() -> bool:
    try:
        return fb.available()[0] and os.path.isfile(tfb.FIXTURE)
    except Exception:
        return False


needs_foul_play = pytest.mark.skipif(not real_foul_play(), reason="Foul Play is not installed here (bench/install_foulplay.sh)")


def real_bridge(fake_ssh: str, helpers: int = 1, beat_s: float = 4.0, **kw) -> FoulPlayBridge:
    kw.setdefault("timeout", 30.0)
    return FoulPlayBridge(helpers=helpers, search_ms=30, remote=target(fake_ssh), env={"FP_BRIDGE_BEAT_TIMEOUT": str(beat_s)}, **kw)


def first_battle():
    events = tfb.load_fixture()[0]
    user, room = events[-1]["username"], events[0]["room"]
    return events, user, room, user + "|" + room


@needs_foul_play
def test_the_real_helper_there_decides_recorded_battles_with_the_round_trip(fake_ssh, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_DELAY_MS", "50")
    bridge = real_bridge(fake_ssh, helpers=2).start()
    try:
        assert "on tester@box" in bridge.install and 95 <= bridge.rtt_start["rtt_ms"] <= 250
        events, user, room, key = first_battle()
        decided = 0
        for event in events:
            if event["t"] == "msg":
                bridge.feed(key, user, room, event["data"])
            elif event["t"] == "order":
                if decided < 4:
                    advice = bridge.advice_sync(key, rqid=event["rqid"], timeout=5.0)
                    recorded = {o["choice"] for o in event["fp_reply"]["options"]}
                    assert {o.choice for o in advice.options} == recorded and advice.iterations > 0
                    assert advice.wall_ms >= advice.ms + 95                # the network, there and back
                    decided += 1
                bridge.note(key, event["choice"])
        assert decided == 4
        info = bridge._helpers[0].info
        assert info["ppid"] != os.getpid() and info["pgid"] == info["pid"]   # its own session and group there
    finally:
        bridge.close()
    assert fb.remote_leftovers(target(fake_ssh)) == []


@needs_foul_play
def test_the_real_helper_there_leaves_when_its_ssh_session_ends(fake_ssh):
    bridge = real_bridge(fake_ssh).start()
    try:
        pid = helper_pid(bridge)
        session = bridge._helpers[0].proc
        # Stop this end's ssh only (no kill by tag): the session there ends, the helper loses its parent.
        session.fp_remote = None
        tfb.kill_hard(session.pid)
        assert tfb.wait_until(lambda: not tfb.pid_alive(pid), timeout=10)
        assert tfb.wait_until(lambda: bridge._helpers[0] is not None and bridge._helpers[0].generation == 2
                              and bridge._helpers[0].replies > 0, timeout=60)
        assert helper_pid(bridge) != pid
    finally:
        bridge.close()


@needs_foul_play
def test_the_real_helper_there_leaves_when_the_connection_goes_quiet(fake_ssh, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_HALF_OPEN_AFTER_S", "12")         # after Foul Play has loaded there
    monkeypatch.setenv("FAKE_SSH_ALIVE_S", "60")                   # this end notices nothing for a minute
    bridge = real_bridge(fake_ssh, beat_s=4.0, hang_timeout=120.0).start(timeout=11.0)
    try:
        pid = helper_pid(bridge)
        assert tfb.pid_alive(pid)
        # Nothing arrives there from 12 s on, and the session there stays open (sshd has not noticed):
        # the helper leaves 4 s (its beat timeout here) after the last line it got.
        assert tfb.wait_until(lambda: not tfb.pid_alive(pid), timeout=12 + 4 + 6)
    finally:
        bridge.close()


@needs_foul_play
def test_calibration_runs_there(fake_ssh):
    result = fb.calibrate(ms_list=(30,), helpers=1, battles=1, remote=target(fake_ssh), timeout=30.0)
    row = result["rows"][0]
    assert row["decisions"] > 0 and row["iterations_per_decision"] > 0 and not result["problems"]
    assert fb.remote_leftovers(target(fake_ssh)) == []


# ---------------------------------------------------------------------------------------------
# ladder.py and the launcher
# ---------------------------------------------------------------------------------------------

def remote_launch_argv(prefix: str):
    """A ladder.py command of cloud/remote_search.sh (one line), with its variables put in."""
    with open(REMOTE_SH, encoding="utf-8") as fh:
        text = fh.read()
    start = text.index(prefix)
    line = text[start:text.index("\n", start)]
    values = {"GAMES": "30", "TAG": "ab3-net-mix-deep-024381", "HOST": "ubuntu@1.2.3.4", "KEY": "C:/k.pem", "KH": "C:/kh",
              "SSH": "C:/Program Files/Git/usr/bin/ssh.exe", "REMOTE_DIR": "~/dev/ab-ladder", "CKPT": "C:/ck/learner_update_024381.pt"}
    for name, value in values.items():
        line = line.replace('"$%s"' % name, '"%s"' % value)
    words = [w[1:-1] if w.startswith('"') and w.endswith('"') else w for w in re.findall(r'"[^"]*"|\S+', line)]
    argv = words[words.index("ladder.py") + 1:]
    assert not any("$" in w for w in argv), line
    return text, argv


def test_the_remote_launcher_plays_the_ab_with_the_box_settings_and_the_search_there():
    import ladder

    text, argv = remote_launch_argv('"$PY" ladder.py --games "$GAMES" --arms net,mix,mix-deep --experiment "$TAG" --predictions --parallel 6')
    args = ladder.parse_args(argv)
    assert args.arms == ["net", "mix", "mix-deep"] and ladder.experiment_tag(args) == "ab3-net-mix-deep-024381"
    assert (args.parallel, args.helpers, args.search_ms, args.deadline, args.predictions) == (6, 7, 500, 2.5, True)
    assert (args.deep_search_ms, args.deep_worlds, args.deep_workers, args.deep_deadline) == (1425, 2, 4, 5.0)
    assert args.search_remote == "ubuntu@1.2.3.4" and args.search_key == "C:/k.pem" and args.search_known_hosts == "C:/kh"
    assert args.search_ssh.endswith("ssh.exe") and args.search_remote_dir == "~/dev/ab-ladder"
    assert args.checkpoint.endswith("learner_update_024381.pt") and args.server == "public" and args.timer
    assert ladder.search_problems(args) == [] and ladder.search_problems(args, rtt_s=0.3) == []
    assert any("round trip" in p for p in ladder.search_problems(args, rtt_s=1.0))     # mix: 2.5 s cannot hold 1 s more
    # The fallback without the box: Look Play AB.cmd's own settings, the search in the laptop's WSL.
    _, argv = remote_launch_argv('"$PY" ladder.py --games "$GAMES" --arms net,mix,mix-deep --experiment "$TAG" --predictions --parallel 3')
    old = ladder.parse_args(argv)
    assert (old.parallel, old.helpers, old.deep_search_ms, old.deep_deadline, old.search_remote) == (3, 4, 1250, 7.5, None)
    assert ladder.search_problems(old) == []
    # The order of the steps, and what it never does.
    steps = text[text.index("# ---- the steps"):]
    order = ["laptop_bots", "free_gb", "start_box", "play_on_laptop", "allow-ip", "remote_ladders", "$(box_training)", "synced_from",
             "remote-check", "ask_account", '"$PY" ladder.py --games', "\n  report\n", "Stop the cloud box now? [Y/n]"]
    at = [steps.index(word) for word in order]
    assert at == sorted(at), list(zip(order, at))
    last = steps.index("Stop the cloud box now? [Y/n]")
    assert steps.index("*) stop_box ;;", last) > last and "n|N|no|No) stop_guard; say" in steps[last:]
    # The box stops itself when the question never comes (the window closed): the guard starts right
    # before ladder.py and is stopped when the owner keeps the box.
    assert steps.index("start_guard") < steps.index('"$PY" ladder.py --games') and "idle_guard.sh ${IDLE_GUARD_MIN:-30}" in text
    with open(os.path.join(REPO, "cloud", "idle_guard.sh")) as fh:
        guard = fh.read()
    assert "sudo -n shutdown -h now" in guard and "tag fpr[.]" in guard and "psppo-train" in guard and "who" in guard
    assert not re.search(r"(^|[;&|(]\s*)rm\s", guard, re.M) and "terminate" not in guard.replace("not terminated", "")
    assert "InsufficientInstanceCapacity" in text and "CAPACITY_WAIT_S=${REMOTE_SEARCH_CAPACITY_WAIT_S:-900}" in text  # 15 minutes
    assert 'read -r -t 60 -p "Stop the cloud box now? [Y/n] "' in text and "stop-instances" in text
    assert "PS_PASSWORD=" not in text and "unset PS_PASSWORD" in text and "read -s" not in text   # the password: ladder.py's prompt only
    assert not re.search(r"(^|[;&|(]\s*)rm\s", text, re.M) and "--clean" not in text            # nothing deleted, here or there
    assert "ProxyCommand" not in text and "ProxyJump" not in text and " -D " not in text         # nothing routed through the box
    assert "--mix-iterations-ref 538000" in text and "-lt 4" in text                              # the report; memory warns under 4 GB
    with open(LAUNCHER, "rb") as fh:
        cmd = fh.read().decode("utf-8")
    assert "cloud/remote_search.sh" in cmd and "cd /d C:\\Users\\you\\ps-ppo-predict" in cmd and "\r\n" in cmd


def fake_box_dir(scratch, start_answer: str) -> str:
    """A cloud/ folder for remote_search.sh: vm.sh says the box is stopped, aws2.sh logs its calls and answers start-instances."""
    vmdir = scratch / "cloud"
    (vmdir / ".state").mkdir(parents=True)
    (vmdir / "vm.sh").write_text('case "$1" in\n  info) echo "i-0abc g6e.16xlarge stopped None us-east-1c None"; echo "spot price now: $1";;\n'
                                 '  *) echo "no running box" >&2; exit 1;;\nesac\n')
    (vmdir / ".state" / "aws2.sh").write_text('echo "$*" >> "%s"\n%s\n' % (scratch / "aws.log", start_answer))
    return str(vmdir)


@pytest.mark.parametrize("choice", ["q", "w"])
def test_the_launcher_waits_for_capacity_then_offers_the_laptop_or_quit(scratch, choice):
    vmdir = fake_box_dir(scratch, 'echo "aws: [ERROR]: An error occurred (InsufficientInstanceCapacity) when calling the '
                                   'StartInstances operation: Insufficient capacity." >&2; exit 254')
    ckpt = scratch / "learner_update_024381.pt"
    ckpt.write_bytes(b"x")
    env = dict(os.environ, PSPPO_VMDIR=vmdir, PSPPO_PY=sys.executable, AB_CKPT=str(ckpt), REMOTE_SEARCH_DRY="1",
               REMOTE_SEARCH_CAPACITY_WAIT_S="2", REMOTE_SEARCH_RETRY_S="1")
    done = subprocess.run(["bash", REMOTE_SH], input=choice + "\n\n", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=120)
    out = done.stdout
    calls = (scratch / "aws.log").read_text().splitlines()
    starts = [c for c in calls if "start-instances" in c]
    assert len(starts) >= 2 and all("--instance-ids i-0abc" in c and "--region us-east-1" in c for c in starts), calls
    assert not any("stop-instances" in c for c in calls)
    assert "trying again in 1 s" in out and "AWS had no capacity for the box" in out and "Q  quit" in out, out
    if choice == "q":
        assert done.returncode == 1 and "dry run" not in out
    else:
        # The laptop's own WSL helpers, as before: Look Play AB.cmd's settings, nothing on the box.
        assert done.returncode == 0, out
        line = next(l for l in out.splitlines() if l.startswith("dry run:"))
        assert "--parallel 3 --helpers 4" in line and "--deep-search-ms 1250" in line and "--search-remote" not in line


def test_the_launcher_stops_at_an_error_that_is_not_capacity(scratch):
    vmdir = fake_box_dir(scratch, 'echo "aws: [ERROR]: An error occurred (UnauthorizedOperation)" >&2; exit 254')
    ckpt = scratch / "c.pt"
    ckpt.write_bytes(b"x")
    env = dict(os.environ, PSPPO_VMDIR=vmdir, PSPPO_PY=sys.executable, AB_CKPT=str(ckpt), REMOTE_SEARCH_DRY="1",
               REMOTE_SEARCH_CAPACITY_WAIT_S="30", REMOTE_SEARCH_RETRY_S="1")
    done = subprocess.run(["bash", REMOTE_SH], input="\n", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert done.returncode == 1 and "the box did not start" in done.stdout and "UnauthorizedOperation" in done.stdout
    assert len((scratch / "aws.log").read_text().splitlines()) == 1        # not tried again


def test_the_ladder_flags_for_the_remote_search():
    import ladder

    args = ladder.parse_args(["--arms", "net,mix", "--search-remote", "vm"])
    assert args.search_remote == "vm" and args.search_remote_dir == "~/dev/ab-ladder" and args.search_key is None
    args = ladder.parse_args(["--arms", "net,mix,mix-deep", "--parallel", "6", "--helpers", "7", "--deep-search-ms", "1425",
                              "--deep-workers", "4", "--deep-deadline", "5.0"])
    assert ladder.search_problems(args, rtt_s=0.25) == []
    problems = ladder.search_problems(args, rtt_s=3.0)
    assert any("--deep-deadline" in p and "round trip" in p for p in problems)


# ---------------------------------------------------------------------------------------------
# the link down is not the search failing (link_status, link_spans, ladder.SearchHealth)
# ---------------------------------------------------------------------------------------------

def test_a_line_the_network_held_up_is_a_link_stall_either_way(fake_ssh, fake_helper):
    import concurrent.futures

    bridge = remote_bridge(fake_ssh, fake_helper)          # not started: its bookkeeping only
    try:
        bridge.rtt_start = {"rtt_ms": 200.0}
        helper = fb._Helper(bridge, 0, 1)
        helper.clock_ok, helper.clock_offset = True, 1000.0  # its clock reads 1000 s more
        now = time.monotonic()

        def pending(sent):
            return fb._Pending(1, {"op": "ping"}, concurrent.futures.Future(), "probe", None, sent)

        bridge._transit(helper, pending(now - 0.3), {"clock": now + 1000.0 - 0.1, "came": now + 1000.0 - 0.2})
        assert bridge.link_status() is None and bridge.link_spans() == []          # a usual line
        bridge._transit(helper, pending(now - 2.0), {"clock": now + 1000.0 - 1.5, "came": now + 1000.0 - 1.9})
        status = bridge.link_status()                                              # held up 1.5 s on its way back
        assert status is not None and "held a line" in status["why"] and status["slots"] == 0
        (start, end), = bridge.link_spans()
        assert end is not None and 1.3 < end - start < 1.7
        bridge._transit(helper, pending(now - 3.0), {"clock": now + 1000.0 - 0.1, "came": now + 1000.0 - 0.5})
        assert len(bridge.link_spans()) == 2 and 2.3 < bridge.link_spans()[1][1] - bridge.link_spans()[1][0] < 2.7  # on its way there
        # A question that waited behind a search there: the helper last waited for a line before it was sent.
        bridge._transit(helper, pending(now - 3.0), {"clock": now + 1000.0 - 0.1, "came": now + 1000.0 - 4.0})
        assert len(bridge.link_spans()) == 2 and bridge.counters["link_late"] == 2
        bridge._late_until = 0.0
        assert bridge.link_status() is None                                        # REMOTE_LATE_HOLD_S later
    finally:
        bridge.close()


def test_an_exit_is_the_links_unless_the_helper_there_failed_by_itself(fake_ssh, fake_helper):
    bridge = remote_bridge(fake_ssh, fake_helper, helpers=5)    # not started: its bookkeeping only

    class Proc:
        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

    try:
        for index, (code, hung, link) in enumerate([(255, False, True),    # ssh lost the connection
                                                    (-9, False, True),     # ssh killed here
                                                    (0, False, True),      # it left there: its parent went quiet
                                                    (7, True, True),       # stopped by the watchdog: it answered nothing
                                                    (7, False, False)]):   # Foul Play's process there failed by itself
            helper = fb._Helper(bridge, index, 1)
            helper.proc, helper.hung = Proc(code), hung
            with bridge._lock:
                bridge._exit_cause(helper, [])
            assert bridge._slot_cause[index][0] == ("link" if link else "helper"), (code, hung)
            assert (bridge._slot_span[index] is not None) == link
        status = bridge.link_status()
        assert status is not None and status["slots"] == 4 and "being replaced" in status["why"]
    finally:
        bridge.close()


def test_a_dropped_link_is_down_until_the_helper_is_back(fake_ssh, fake_helper):
    bridge = remote_bridge(fake_ssh, fake_helper, helpers=2).start()
    try:
        assert bridge.link_status() is None and bridge.link_spans() == []
        old = bridge._helpers[0].proc.pid
        dropped = time.time()
        tfb.kill_hard(old)                                     # its ssh goes: the connection is lost
        assert tfb.wait_until(lambda: bridge.link_status() is not None, timeout=5)
        status = bridge.link_status()
        assert "being replaced" in status["why"] and dropped - 0.1 <= status["since"] <= dropped + 0.6, status
        assert tfb.wait_until(lambda: bridge.link_status() is None, timeout=30)   # the new one answers
        (start, end), = bridge.link_spans()
        assert end is not None and start <= dropped + 0.6 and end > start and bridge.helpers_up() == 2
    finally:
        bridge.close()


def test_a_helper_that_answers_nothing_is_down_from_when_it_went_quiet(fake_ssh, fake_helper):
    bridge = remote_bridge(fake_ssh, fake_helper, timeout=1.0, hang_timeout=3.0).start()
    try:
        tfb.feed_request(bridge, "u|battle-hang", rqid=1)
        asked = time.time()
        bridge.advice_future("u|battle-hang", rqid=1)          # answers nothing for an hour
        assert tfb.wait_until(lambda: bridge.link_status() is not None, timeout=5)
        status = bridge.link_status()
        assert "answered nothing" in status["why"] and asked - 0.1 <= status["since"] <= asked + 0.5, status
        assert time.time() - asked >= 1.0                      # down after `timeout`, dated back to the question
        assert tfb.wait_until(lambda: bridge.counters["hangs"] >= 1, timeout=10)
        # Replaced; the question is asked once more of the new helper, which hangs on it too, and then given up.
        assert tfb.wait_until(lambda: bridge.counters["hangs"] == 2 and bridge.link_status() is None, timeout=40)
        spans = bridge.link_spans()
        assert spans and asked - 0.1 <= spans[0][0] <= asked + 0.5 and all(end is not None for _, end in spans), spans
    finally:
        bridge.close()


def test_failures_while_the_link_was_down_leave_the_health_window_and_real_ones_stay():
    import ladder

    health = ladder.SearchHealth(50, settle_s=7.0)
    t0 = 1_000_000.0
    for i in range(45):
        health.note({"mode": "mix", "reason": "", "ms": 1400}, now=t0 + i)
    for i in range(12):                                         # a 7 s cut: 12 fell back
        health.note({"mode": "mix-deep" if i % 2 else "mix", "reason": ("timeout", "down", "error")[i % 3], "ms": 2450},
                    now=t0 + 50 + 0.5 * i)
    assert health.share(now=t0 + 70) is not None and health.share(now=t0 + 70) > 0.10          # as before: it would stop
    assert health.share(now=t0 + 54) is None or health.share(now=t0 + 54) < 0.10            # not judged before settle_s
    assert health.link_spans([(t0 + 49.9, t0 + 57.0)]) == 12
    assert health.failed == 0 and health.link_failed == 12 and health.total == 57
    assert health.share(now=t0 + 70) is None                                                 # 45 left in the window
    for i in range(5):
        health.note({"mode": "mix", "reason": "", "ms": 1400}, now=t0 + 60 + i)
    assert health.share(now=t0 + 80) == 0.0 and "not counted" in health.text(now=t0 + 80)
    # Real failures on a healthy link: still the rule's.
    for i in range(6):
        health.note({"mode": "mix", "reason": "error", "ms": 30}, now=t0 + 70 + i)
    assert health.link_spans([(t0 + 49.9, t0 + 57.0), (t0 + 200.0, None)]) == 0
    assert health.share(now=t0 + 90) == 6 / 50.0
    # An open span (the link is down now) takes what fails during it.
    health.note({"mode": "mix", "reason": "timeout", "ms": 2450}, now=t0 + 202.0)
    assert health.link_spans([(t0 + 200.0, None)]) == 1
    # Without a remote search (settle_s 0) nothing changes: judged at once, a window of 50.
    plain = ladder.SearchHealth(50)
    for i in range(50):
        plain.note({"mode": "mix", "reason": "down" if i < 6 else ""})
    assert plain.share() == 6 / 50.0


def test_the_ladder_flag_for_the_link_down_limit():
    import ladder

    assert ladder.parse_args(["--arms", "net,mix", "--search-remote", "vm"]).link_down_limit == 180.0
    assert ladder.parse_args(["--link-down-limit", "60"]).link_down_limit == 60.0
