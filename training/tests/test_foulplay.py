"""
Tests for bench/foulplay.py that need neither Foul Play nor a Showdown server.

    python -m pytest tests/test_foulplay.py -q

What stands in for the real things:
  * a fake Foul Play checkout: a folder with run.py, fp/ and the set data file, whose fp/main.py
    is a small script that "plays" by exchanging files with the test (see FAKE_MAIN). It is
    started by the real code path, `python bench/foulplay.py --child <dir> ...`, so the child
    side of the module (path set-up, the login patch) is exercised too.
  * a fake player with the attributes run_matches uses from a poke-env 0.15 Player.

The real thing (Foul Play, poke-engine, a Showdown server, poke-env players) was tested in WSL:
the commands and results are in bench/FOULPLAY.md.

Nothing here deletes anything. Temporary folders are made under $FOULPLAY_TEST_TMP (default: the
system temp folder) and left there. Every process is started by bench/foulplay.py itself and
stopped by it, by PID.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bench import foulplay  # noqa: E402

PLAYER = "tester"
SHA = "0123456789abcdef0123456789abcdef01234567"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def new_dir(prefix: str) -> str:
    base = os.environ.get("FOULPLAY_TEST_TMP") or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie that nobody waited for still answers signal 0
    try:
        with open("/proc/%d/stat" % pid) as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


# The fake Foul Play. One process plays --run-count games, one after the other, like the real one.
# Game k of process NAME:   it writes   <box>/NAME.challenge.k
#                           the test    <box>/NAME.accepted.k   (holding the battle tag)
#                           it writes   <box>/NAME.result.k     (holding the winner's name, or None)
FAKE_MAIN = r'''
import asyncio
import os
import subprocess
import sys


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def put(box, name, text):
    tmp = os.path.join(box, name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
    os.rename(tmp, os.path.join(box, name))


async def take(box, name):
    path = os.path.join(box, name)
    while True:
        try:
            with open(path) as fh:
                return fh.read()
        except OSError:
            # not there yet, or (Windows) still being renamed into place by the writer
            await asyncio.sleep(0.02)


async def run_foul_play():
    from fp.websocket_client import PSWebsocketClient

    name = arg("--ps-username")
    player = arg("--user-to-challenge")
    count = int(arg("--run-count", "1"))
    scenario = os.environ.get("FAKE_FP_SCENARIO", "play")
    box = os.environ["FAKE_FP_MAILBOX"]
    outcomes = os.environ.get("FAKE_FP_OUTCOMES", "L")
    game_seconds = float(os.environ.get("FAKE_FP_GAME_SECONDS", "0.15"))
    attempt = int(name.rsplit("r", 1)[1])
    print("INFO     login patched: %s" % (PSWebsocketClient.login.__name__ == "_local_login"), flush=True)
    print("INFO     argv: %s" % " ".join(sys.argv[1:]), flush=True)
    print("INFO     cwd: %s" % os.getcwd(), flush=True)
    print("INFO     parent pid: %s" % os.environ.get("FOULPLAY_PARENT_PID"), flush=True)
    print("INFO     pythonpath: %r" % os.environ.get("PYTHONPATH"), flush=True)
    if scenario == "crash_at_start":
        raise RuntimeError("boom at start")
    if scenario == "hang_before_challenge":
        await asyncio.sleep(3600)
    if scenario == "grandchild":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        put(box, name + ".grandchild", str(child.pid))
        await asyncio.sleep(3600)
    for k in range(count):
        put(box, "%s.challenge.%d" % (name, k), "")
        print("INFO     Challenging %s..." % player, flush=True)
        tag = await take(box, "%s.accepted.%d" % (name, k))
        print("INFO     Initialized %s against: %s" % (tag, player), flush=True)
        if scenario == "crash_in_game" or (scenario == "crash_once" and attempt == 0 and k == 1):
            raise RuntimeError("engine exploded")
        if scenario == "hang_in_game":
            await asyncio.sleep(3600)
        print("INFO     Searching for a move using MCTS...", flush=True)
        print("INFO     Iterations 0: 1000", flush=True)
        print("INFO     Iterations 0: 1000", flush=True)
        await asyncio.sleep(game_seconds)
        outcome = outcomes[int(tag.rsplit("-", 1)[1]) % len(outcomes)]
        winner = {"W": player, "L": name, "T": "None"}[outcome]
        print("INFO     Winner: %s" % winner, flush=True)
        put(box, "%s.result.%d" % (name, k), winner)
    print("INFO     W: 0\tL: 0", flush=True)
'''


def make_checkout(with_venv_stubs: bool = True, with_sets: bool = True, head: str = SHA) -> str:
    """A folder that passes for a Foul Play installation."""
    root = new_dir("fakefp_")
    write(os.path.join(root, "run.py"), "raise SystemExit('the benchmark does not start run.py directly')\n")
    write(os.path.join(root, "fp", "__init__.py"), "")
    write(os.path.join(root, "fp", "websocket_client.py"),
          "class PSWebsocketClient:\n    async def login(self):\n        return 'stock login'\n")
    write(os.path.join(root, "fp", "main.py"), FAKE_MAIN)
    if with_sets:
        write(os.path.join(root, "fp", "data", "pkmn_sets_cache", "gen9randombattle.json"),
              json.dumps({"pokemon%d" % i: {"80,item,ability,a,b,c,d,fire": 1} for i in range(400)}))
    if with_venv_stubs:
        # what `import poke_engine, websockets, requests, dateutil` finds from the checkout
        for module in ("poke_engine", "websockets", "requests", "dateutil"):
            write(os.path.join(root, module + ".py"), "")
    write(os.path.join(root, ".git", "HEAD"), head + "\n")
    return root


class FakeBattle:
    def __init__(self, opponent: str):
        self.opponent_username = opponent
        self.finished = False
        self.won = None


class FakeClient:
    def __init__(self, player: "FakePlayer"):
        self.player = player
        self.loop = asyncio.get_running_loop()
        self.logged_in = asyncio.Event()
        self.logged_in.set()
        self.sent = []

    async def accept_challenge(self, username: str, team):
        self.sent.append(("/accept", username))
        self.player.accept(username)

    async def send_message(self, message: str, room: str = ""):
        self.sent.append((message, room))
        if message == "/forfeit" and room in self.player.battles:
            battle = self.player.battles[room]
            battle.won = False
            battle.finished = True


class FakePlayer:
    """The attributes of a poke-env 0.15 Player that run_matches uses. Build it inside a running loop."""

    def __init__(self, mailbox: str, name: str = PLAYER, max_concurrent_battles: int = 0, duplicate_challenges: bool = False):
        self.username = name
        self.format = "gen9randombattle"
        self.mailbox = mailbox
        self.battles = {}
        self._max_concurrent_battles = max_concurrent_battles
        self._challenge_queue = asyncio.Queue()
        self._battle_semaphore = asyncio.Semaphore(0)
        self.ps_client = FakeClient(self)
        self.duplicate_challenges = duplicate_challenges
        self.accept_creates_battle = True
        self._seen = set()
        self._next_game = {}     # process name -> index of the game its next accept starts
        self._battle_of = {}     # (name, k) -> tag
        self._counter = 0
        self._task = asyncio.ensure_future(self._poll())

    def get_next_team(self):
        return None

    def accept(self, username: str) -> None:
        if not self.accept_creates_battle:
            return
        k = self._next_game.get(username, 0)
        self._next_game[username] = k + 1
        tag = "battle-gen9randombattle-%d" % self._counter
        self._counter += 1
        self.battles[tag] = FakeBattle(username)
        self._battle_of[(username, k)] = tag
        self._battle_semaphore.release()
        tmp = os.path.join(self.mailbox, "%s.accepted.%d.tmp" % (username, k))
        with open(tmp, "w") as fh:
            fh.write(tag)
        os.rename(tmp, os.path.join(self.mailbox, "%s.accepted.%d" % (username, k)))

    async def _poll(self) -> None:
        while True:
            for entry in sorted(os.listdir(self.mailbox)):
                if entry in self._seen or entry.endswith(".tmp"):
                    continue
                parts = entry.split(".")
                if len(parts) != 3:
                    continue
                name, kind, k = parts[0], parts[1], int(parts[2])
                if kind == "challenge":
                    self._seen.add(entry)
                    self._challenge_queue.put_nowait(name)
                    if self.duplicate_challenges:
                        self._challenge_queue.put_nowait(name)
                elif kind == "result":
                    try:
                        with open(os.path.join(self.mailbox, entry)) as fh:
                            winner = fh.read().strip()
                    except OSError:
                        continue  # Windows: the writer is still renaming it into place
                    self._seen.add(entry)
                    battle = self.battles[self._battle_of[(name, k)]]
                    battle.won = True if winner == self.username else (None if winner == "None" else False)
                    battle.finished = True
            await asyncio.sleep(0.02)

    async def close(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


@pytest.fixture
def fake(monkeypatch):
    """A fake installation, a mailbox and fast timings. Yields a small namespace."""
    checkout = make_checkout()
    mailbox = new_dir("fpbox_")
    logs = new_dir("fplogs_")
    monkeypatch.setenv("FOULPLAY_DIR", checkout)
    monkeypatch.setenv("FOULPLAY_PYTHON", sys.executable)
    monkeypatch.setenv("FAKE_FP_MAILBOX", mailbox)
    monkeypatch.setenv("FAKE_FP_SCENARIO", "play")
    monkeypatch.setenv("FAKE_FP_OUTCOMES", "L")
    monkeypatch.setattr(foulplay, "_POLL_S", 0.05)
    monkeypatch.setattr(foulplay, "_STAGGER_S", 0.05)
    monkeypatch.setattr(foulplay, "_ACCEPT_WAIT_S", 2.0)
    monkeypatch.setattr(foulplay, "_ABORT_WAIT_S", 2.0)
    monkeypatch.setattr(foulplay, "_SETTLE_WAIT_S", 0.5)

    class Namespace:
        pass

    ns = Namespace()
    ns.checkout, ns.mailbox, ns.logs, ns.env = checkout, mailbox, logs, monkeypatch
    yield ns
    # whatever a failing test left behind: stopped by the module, by the PIDs it started
    foulplay._kill_everything()
    assert not foulplay._LIVE


def play(ns, n_games: int, **kwargs):
    """Runs run_matches against the fake on one event loop. Returns (result, player)."""
    player_kwargs = kwargs.pop("player_kwargs", {})

    async def go():
        player = FakePlayer(ns.mailbox, **player_kwargs)
        try:
            result = await foulplay.run_matches(player, n_games, "127.0.0.1", 1, log_dir=ns.logs, **kwargs)
        finally:
            await player.close()
        return result, player

    result, player = asyncio.run(go())
    dump = os.environ.get("FOULPLAY_TEST_DUMP")
    if dump:  # for chasing a flaky run: everything that happened, one JSON object per call
        with open(dump, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"n_games": n_games, "kwargs": {k: repr(v) for k, v in kwargs.items()},
                                 "result": result, "sent": player.ps_client.sent,
                                 "battles": {tag: [b.opponent_username, b.finished, b.won] for tag, b in player.battles.items()},
                                 "mailbox": sorted(os.listdir(ns.mailbox)), "logs": logs_text(ns)}))
            fh.write("\n")
    return result, player


def logs_text(ns) -> str:
    text = []
    for entry in sorted(os.listdir(ns.logs)):
        with open(os.path.join(ns.logs, entry), encoding="utf-8", errors="replace") as fh:
            text.append(fh.read())
    return "\n".join(text)


def assert_invariants(result, n_games: int) -> None:
    assert result["games"] == result["wins"] + result["losses"] + result["ties"]
    assert result["games"] + result["errors"] == n_games
    assert result["requested"] == n_games
    assert not foulplay._LIVE, "a process was left running"
    for key in ("games", "wins", "losses", "ties", "errors", "seconds", "foulplay_commit", "search_time_ms", "parallel"):
        assert key in result


# ---------------------------------------------------------------------------------------------
# available()
# ---------------------------------------------------------------------------------------------

def test_available_when_nothing_is_installed(monkeypatch):
    missing = os.path.join(new_dir("nofp_"), "foul-play")
    monkeypatch.setenv("FOULPLAY_DIR", missing)
    ok, why = foulplay.available()
    assert ok is False
    assert missing in why and "install_foulplay.sh" in why


def test_available_default_directory_is_home_foul_play(monkeypatch):
    monkeypatch.delenv("FOULPLAY_DIR", raising=False)
    assert foulplay.foulplay_dir() == os.path.abspath(os.path.expanduser("~/foul-play"))


def test_available_rejects_a_folder_that_is_not_foul_play(monkeypatch):
    monkeypatch.setenv("FOULPLAY_DIR", new_dir("empty_"))
    ok, why = foulplay.available()
    assert ok is False and "not a Foul Play checkout" in why


def test_available_needs_the_virtualenv(monkeypatch):
    monkeypatch.setenv("FOULPLAY_DIR", make_checkout())
    monkeypatch.delenv("FOULPLAY_PYTHON", raising=False)
    ok, why = foulplay.available()
    assert ok is False and "no virtualenv" in why


def test_available_needs_the_set_data(monkeypatch):
    monkeypatch.setenv("FOULPLAY_DIR", make_checkout(with_sets=False))
    monkeypatch.setenv("FOULPLAY_PYTHON", sys.executable)
    ok, why = foulplay.available()
    assert ok is False and "set data" in why


def test_available_when_the_engine_cannot_be_imported(monkeypatch):
    monkeypatch.setenv("FOULPLAY_DIR", make_checkout(with_venv_stubs=False))
    monkeypatch.setenv("FOULPLAY_PYTHON", sys.executable)
    ok, why = foulplay.available()
    assert ok is False
    assert "cannot import its requirements" in why and "poke_engine" in why


def test_available_reports_commit_and_directory(monkeypatch):
    checkout = make_checkout(head=foulplay.PINNED_COMMIT)
    monkeypatch.setenv("FOULPLAY_DIR", checkout)
    monkeypatch.setenv("FOULPLAY_PYTHON", sys.executable)
    assert foulplay.available() == (True, "foul-play %s at %s" % (foulplay.PINNED_COMMIT, os.path.abspath(checkout)))


def test_available_says_when_the_commit_is_not_the_pinned_one(monkeypatch):
    monkeypatch.setenv("FOULPLAY_DIR", make_checkout(head=SHA))
    monkeypatch.setenv("FOULPLAY_PYTHON", sys.executable)
    ok, why = foulplay.available()
    assert ok is True
    assert why.startswith("foul-play %s at " % SHA) and "not the pinned commit" in why


def test_available_never_raises(monkeypatch):
    def explode():
        raise OSError("disk on fire")

    monkeypatch.setattr(foulplay, "_locate", explode)
    ok, why = foulplay.available()
    assert ok is False and "disk on fire" in why


def test_read_commit_from_a_branch_and_from_packed_refs():
    root = new_dir("git_")
    write(os.path.join(root, ".git", "HEAD"), "ref: refs/heads/main\n")
    assert foulplay._read_commit(root) == "unknown"
    write(os.path.join(root, ".git", "packed-refs"), "# pack-refs with: peeled\n%s refs/heads/main\n" % SHA)
    assert foulplay._read_commit(root) == SHA
    other = "f" * 40
    write(os.path.join(root, ".git", "refs", "heads", "main"), other + "\n")
    assert foulplay._read_commit(root) == other
    assert foulplay._read_commit(os.path.join(root, "nowhere")) == "unknown"


# ---------------------------------------------------------------------------------------------
# small pure pieces
# ---------------------------------------------------------------------------------------------

def test_split_games():
    assert foulplay._split_games(10, 4) == [3, 3, 2, 2]
    assert foulplay._split_games(100, 8) == [13, 13, 13, 13, 12, 12, 12, 12]
    assert foulplay._split_games(2, 8) == [1, 1]
    assert foulplay._split_games(5, 1) == [5]
    assert foulplay._split_games(0, 4) == []
    for n in range(1, 40):
        for slots in range(1, 12):
            shares = foulplay._split_games(n, slots)
            assert sum(shares) == n and min(shares) >= 1 and max(shares) - min(shares) <= 1


def test_to_id_matches_showdown():
    assert foulplay._to_id(" FPab12s0r0@!") == "fpab12s0r0"
    assert foulplay._to_id("Random Player 1") == "randomplayer1"


def test_game_timeout_grows_with_search_time():
    assert foulplay._default_game_timeout(200) == 240.0
    assert foulplay._default_game_timeout(5000) > 2000.0


def test_log_lines_are_understood():
    proc = foulplay._Proc(0, 0, "FPx", 3, popen=None, log_path="none.log")
    for line in [
        "INFO     Logged in as FPx without an assertion (local server)",
        "INFO     Searching for a move using MCTS...",
        "INFO     Iterations 0: 120000",
        "INFO     Iterations 1: 80000",
        "INFO     Winner: FPx",
        "INFO     Winner: tester",
        "INFO     Winner: None",
        "WARNING  ghost is not a known move",
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "    raise ValueError('bad state')",
        "ValueError: bad state",
        "INFO     Usage: cpu_s=1.5 worker_cpu_s=20.5 rss_mb=50 worker_rss_mb=120",
    ]:
        proc.parse_line(line)
    assert proc.winners == ["FPx", "tester", "None"]
    assert proc.searches == 1
    assert (proc.iterations_sum, proc.iterations_n) == (200000, 2)
    assert proc.exception == "ValueError: bad state"
    assert proc.why() == "ValueError: bad state"
    assert proc.cpu_seconds == 22.0 and proc.rss_mb == 50 and proc.worker_rss_mb == 120


def test_child_environment_drops_the_callers_python_settings(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/training")
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/venv")
    env = foulplay._child_env()
    assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env
    assert env["FOULPLAY_PARENT_PID"] == str(os.getpid())
    assert env["PYTHONUNBUFFERED"] == "1"


def test_local_login_sends_an_empty_assertion_and_waits_for_the_name():
    class Client:
        username = "FPab12s0r0"

        def __init__(self, messages):
            self.messages = list(messages)
            self.sent = []

        async def receive_message(self):
            if not self.messages:
                await asyncio.sleep(3600)
            return self.messages.pop(0)

        async def send_message(self, room, message_list):
            self.sent.append((room, list(message_list)))

    ok = Client(["|updateuser| Guest 12|0|1|{}", "|challstr|4|abcdef", "|updateuser| FPab12s0r0|1|1|{}"])
    assert asyncio.run(foulplay._local_login(ok)) == "FPab12s0r0"
    assert ok.sent == [("", ["/trn FPab12s0r0,0,"])]

    taken = Client(["|challstr|4|abcdef", "|nametaken|FPab12s0r0|Your authentication token was invalid."])
    with pytest.raises(RuntimeError, match="refused the name"):
        asyncio.run(foulplay._local_login(taken))

    silent = Client(["|challstr|4|abcdef"])
    old = foulplay._CHILD_LOGIN_TIMEOUT_S
    foulplay._CHILD_LOGIN_TIMEOUT_S = 0.2
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            asyncio.run(foulplay._local_login(silent))
    finally:
        foulplay._CHILD_LOGIN_TIMEOUT_S = old


# ---------------------------------------------------------------------------------------------
# run_matches: refusals that start nothing
# ---------------------------------------------------------------------------------------------

def test_run_matches_reports_errors_when_foul_play_is_missing(fake):
    fake.env.setenv("FOULPLAY_DIR", os.path.join(new_dir("nofp_"), "foul-play"))
    result, _ = play(fake, 5)
    assert_invariants(result, 5)
    assert result["errors"] == 5 and result["games"] == 0
    assert "not installed" in result["error"]
    assert os.listdir(fake.logs) == []


def test_run_matches_with_no_games(fake):
    result, _ = play(fake, 0)
    assert (result["games"], result["errors"]) == (0, 0)


def test_run_matches_refuses_a_player_it_cannot_use(fake):
    class NotAPlayer:
        pass

    result = asyncio.run(foulplay.run_matches(NotAPlayer(), 3, "127.0.0.1", 1))
    assert result["errors"] == 3 and "poke-env 0.15 Player" in result["error"]

    async def wrong_format():
        player = FakePlayer(fake.mailbox)
        player.format = "gen9ou"
        try:
            return await foulplay.run_matches(player, 3, "127.0.0.1", 1, log_dir=fake.logs)
        finally:
            await player.close()

    result = asyncio.run(wrong_format())
    assert result["errors"] == 3 and "gen9ou" in result["error"]

    result, _ = play(fake, 2, player_kwargs={"name": "a vs. b"})
    assert result["errors"] == 2 and "confuse" in result["error"]
    assert os.listdir(fake.logs) == []


# ---------------------------------------------------------------------------------------------
# run_matches with the fake Foul Play
# ---------------------------------------------------------------------------------------------

def test_games_are_counted_from_the_players_side(fake):
    fake.env.setenv("FAKE_FP_OUTCOMES", "WLT")
    result, player = play(fake, 7, parallel=3)
    assert_invariants(result, 7)
    # battle tags 0..6 with the pattern W L T: W L T W L T W
    assert (result["wins"], result["losses"], result["ties"], result["errors"]) == (3, 2, 2, 0), (result, logs_text(fake)[-3000:])
    assert result["foulplay_log"] == {"wins": 3, "losses": 2, "ties": 2}
    assert result["log_agrees"] is True
    assert result["parallel"] == 3 and result["restarts"] == 0
    assert result["foulplay_commit"] == SHA
    assert result["foulplay_decisions"] == 7 and result["foulplay_mean_iterations"] == 1000
    assert "failures" not in result and "error" not in result
    assert player._battle_semaphore._value == 0, "the player's battle semaphore was left unbalanced"
    assert not any(message == "/forfeit" for message, _room in player.ps_client.sent)
    names = sorted(os.listdir(fake.logs))
    assert len(names) == 3
    run_tag = names[0].split("_")[1]
    assert names == ["foulplay_%s_s%02d_r0.log" % (run_tag, slot) for slot in range(3)]


def test_the_child_is_started_the_way_foul_play_expects(fake):
    result, _ = play(fake, 2, parallel=2, search_time_ms=350, search_parallelism=2)
    assert_invariants(result, 2)
    text = logs_text(fake)
    assert "login patched: True" in text
    working_dirs = [line.split("cwd: ", 1)[1] for line in text.splitlines() if "cwd: " in line]
    assert len(working_dirs) == 2 and all(os.path.samefile(d, fake.checkout) for d in working_dirs)
    assert "parent pid: %d" % os.getpid() in text
    assert "pythonpath: None" in text
    for expected in ("--websocket-uri ws://127.0.0.1:1/showdown/websocket", "--bot-mode challenge_user",
                     "--user-to-challenge %s" % PLAYER, "--pokemon-format gen9randombattle",
                     "--search-time-ms 350", "--search-parallelism 2", "--run-count 1", "--log-level INFO"):
        assert expected in text, expected
    assert "--ps-password" not in text
    assert (result["search_time_ms"], result["search_parallelism"]) == (350, 2)


def test_a_challenge_seen_twice_is_accepted_once(fake):
    result, player = play(fake, 4, parallel=2, player_kwargs={"duplicate_challenges": True})
    assert_invariants(result, 4)
    assert (result["losses"], result["errors"]) == (4, 0)
    assert result["ignored_challenges"] == 4
    assert sum(1 for message, _ in player.ps_client.sent if message == "/accept") == 4


def test_parallel_is_lowered_to_what_the_player_can_take(fake):
    result, _ = play(fake, 3, parallel=3, player_kwargs={"max_concurrent_battles": 1})
    assert_invariants(result, 3)
    assert result["parallel"] == 1 and result["games"] == 3
    assert any("max_concurrent_battles=1" in note for note in result["notes"])
    assert len(os.listdir(fake.logs)) == 1


def test_a_process_that_never_challenges_is_stopped_and_counted(fake):
    fake.env.setenv("FAKE_FP_SCENARIO", "hang_before_challenge")
    t0 = time.monotonic()
    result, _ = play(fake, 3, parallel=2, start_timeout_s=1.0, max_restarts=1)
    assert_invariants(result, 3)
    assert (result["games"], result["errors"]) == (0, 3)
    assert len(result["failures"]) == 3 and all("no battle started within 1 s" in f for f in result["failures"])
    assert result["restarts"] == 1
    assert time.monotonic() - t0 < 30


def test_a_process_that_is_never_accepted_is_stopped_and_counted(fake):
    async def go():
        player = FakePlayer(fake.mailbox)
        player.accept_creates_battle = False   # the server never starts the battle
        try:
            return await foulplay.run_matches(player, 2, "127.0.0.1", 1, log_dir=fake.logs, parallel=2,
                                              start_timeout_s=3.0, max_restarts=0)
        finally:
            await player.close()

    fake.env.setattr(foulplay, "_ACCEPT_WAIT_S", 0.3)
    result = asyncio.run(go())
    assert_invariants(result, 2)
    assert (result["games"], result["errors"]) == (0, 2)
    assert any("accepted but no battle started" in note for note in result["notes"])


def test_a_process_that_dies_at_once_is_an_error_with_its_reason(fake):
    fake.env.setenv("FAKE_FP_SCENARIO", "crash_at_start")
    result, _ = play(fake, 4, parallel=2, max_restarts=1)
    assert_invariants(result, 4)
    assert (result["games"], result["errors"]) == (0, 4)
    assert all("RuntimeError: boom at start" in failure for failure in result["failures"])
    assert all("exited with code 1 after 0 of" in failure for failure in result["failures"])


def test_a_process_that_dies_in_a_game_is_replaced(fake):
    fake.env.setenv("FAKE_FP_SCENARIO", "crash_once")
    result, player = play(fake, 6, parallel=2)
    assert_invariants(result, 6)
    # each of the two slots: game 1 played, game 2 lost to the crash, game 3 by the replacement
    assert (result["games"], result["losses"], result["errors"], result["restarts"]) == (4, 4, 2, 2)
    assert result["log_agrees"] is True
    assert len(result["failures"]) == 2
    assert all("during battle-gen9randombattle-" in f and "RuntimeError: engine exploded" in f for f in result["failures"])
    forfeited = [room for message, room in player.ps_client.sent if message == "/forfeit"]
    assert len(forfeited) == 2
    assert all(player.battles[tag].finished for tag in player.battles), "a battle was left open on the player"
    assert player._battle_semaphore._value == 0


def test_a_game_that_never_ends_is_stopped_and_counted(fake):
    fake.env.setenv("FAKE_FP_SCENARIO", "hang_in_game")
    result, player = play(fake, 2, parallel=2, game_timeout_s=1.0, max_restarts=0)
    assert_invariants(result, 2)
    assert (result["games"], result["errors"]) == (0, 2)
    assert len(result["failures"]) == 2, result
    assert all("was not over after 1 s" in failure for failure in result["failures"]), result["failures"]
    assert sorted(room for message, room in player.ps_client.sent if message == "/forfeit") == sorted(player.battles)
    assert all(battle.finished for battle in player.battles.values())
    # forfeited games are losses on the player object, and are not in the result
    assert result["losses"] == 0


def test_the_overall_timeout_stops_everything(fake):
    fake.env.setenv("FAKE_FP_GAME_SECONDS", "60")
    t0 = time.monotonic()
    result, player = play(fake, 4, parallel=2, overall_timeout_s=1.5)
    assert_invariants(result, 4)
    assert time.monotonic() - t0 < 30
    assert result["timed_out"] is True and (result["games"], result["errors"]) == (0, 4)
    assert any("overall timeout" in failure for failure in result["failures"])
    assert all(battle.finished for battle in player.battles.values())


def test_stopping_a_process_takes_what_it_started_with_it(fake):
    fake.env.setenv("FAKE_FP_SCENARIO", "grandchild")
    result, _ = play(fake, 1, parallel=1, start_timeout_s=2.0, max_restarts=0)
    assert_invariants(result, 1)
    notes = [entry for entry in os.listdir(fake.mailbox) if entry.endswith(".grandchild")]
    assert len(notes) == 1
    with open(os.path.join(fake.mailbox, notes[0])) as fh:
        grandchild = int(fh.read())
    for _ in range(50):
        if not pid_alive(grandchild):
            break
        time.sleep(0.1)
    assert not pid_alive(grandchild), "the process started by the fake Foul Play survived"


def test_cancelling_run_matches_leaves_no_process(fake):
    fake.env.setenv("FAKE_FP_SCENARIO", "hang_in_game")

    async def go():
        player = FakePlayer(fake.mailbox)
        task = asyncio.ensure_future(foulplay.run_matches(player, 2, "127.0.0.1", 1, log_dir=fake.logs, parallel=2))
        for _ in range(400):
            if len(player.battles) == 2:
                break
            await asyncio.sleep(0.05)
        assert len(player.battles) == 2 and len(foulplay._LIVE) == 2
        pids = [popen.pid for popen in foulplay._LIVE]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await player.close()
        return pids, player

    pids, player = asyncio.run(go())
    assert not foulplay._LIVE
    assert not any(pid_alive(pid) for pid in pids)
    assert all(battle.finished for battle in player.battles.values())


def test_a_player_on_another_event_loop_like_poke_env(fake):
    """poke-env keeps its players on a loop in a background thread; the caller awaits from its own."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        async def build():
            return FakePlayer(fake.mailbox)

        player = asyncio.run_coroutine_threadsafe(build(), loop).result(timeout=10)
        fake.env.setenv("FAKE_FP_OUTCOMES", "W")
        result = asyncio.run(foulplay.run_matches(player, 3, "127.0.0.1", 1, log_dir=fake.logs, parallel=2))
        assert_invariants(result, 3)
        assert (result["wins"], result["errors"]) == (3, 0)
        asyncio.run_coroutine_threadsafe(player.close(), loop).result(timeout=10)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)


def test_two_runs_on_one_player_count_only_their_own_games(fake):
    async def go():
        player = FakePlayer(fake.mailbox)
        try:
            first = await foulplay.run_matches(player, 3, "127.0.0.1", 1, log_dir=fake.logs, parallel=2)
            second = await foulplay.run_matches(player, 2, "127.0.0.1", 1, log_dir=fake.logs, parallel=2)
        finally:
            await player.close()
        return first, second, player

    first, second, player = asyncio.run(go())
    assert_invariants(first, 3)
    assert_invariants(second, 2)
    assert (first["games"], second["games"]) == (3, 2)
    assert len(player.battles) == 5 and player._battle_semaphore._value == 0
