"""
ladder.py's login (log_in_checked, note_updateuser, note_server_word, first_login), against a fake
login server and a fake Showdown on this machine (127.0.0.1, ports 10900 to 10999, local names, a
made-up password). Nothing here talks to the public server or its login server.

    python -m pytest tests/test_ladder_login.py -q -p no:cacheprovider

The login server's answers: a good assertion, ";;Wrong password.", actionsuccess false, an HTML
page (a Cloudflare block), no answer in time. Showdown's: a confirmation at once, one 12 s late,
none, |nametaken|, a |popup| about a lock with the lock mark on the name, a |raw| line. In every
case one line says what happened, and neither the password nor the assertion is in anything
printed, in the console log or in the exit file. Also: run_box_ab.sh's line after ladder.py ends,
and make_player connecting only once the login handler is in place.
"""

from __future__ import annotations

import asyncio
import glob
import http.server
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import ladder  # noqa: E402

PASSWORD = "pw-Fake-Secret-913"
ASSERTION = "4,logintest,2,1760000000,sim3.psim.us,ASSERT-SECRET-" + "9f" * 20
CHALLSTR = "4|" + "c0ffee" * 10
NAMES = itertools.count(1)


def free_port() -> int:
    for port in range(10900, 11000):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 10900-10999")


# =============================================================================================
# The login server's answer, read
# =============================================================================================

@pytest.mark.parametrize("status, text, kind, words", [
    (200, ']{"actionsuccess":true,"assertion":"%s","curuser":{"loggedin":true}}' % ASSERTION, None, None),
    (200, ']{"actionsuccess":true,"assertion":";;Wrong password."}', "refused", "(status 200): Wrong password."),
    (200, ']{"actionsuccess":false,"assertion":";;Wrong password."}', "refused", "(status 200): Wrong password."),
    (200, ']{"actionsuccess":false,"assertion":false}', "refused", "no reason given (actionsuccess false)"),
    (200, ']{"actionerror":"Too many unsuccessful login attempts."}', "refused", ": Too many unsuccessful login attempts."),
    (200, ']{"assertion":";"}', "refused", "the name is registered and it did not take the password"),
    (403, "<!DOCTYPE html><html><head><title>Attention Required! | Cloudflare</title></head><body>blocked</body></html>",
     "failed", "the login server answered with a web page, status 403 (likely a block on this address)"),
    (502, "", "failed", "not JSON (0 characters), status 502"),
    (200, ']{"actionsuccess":true}', "failed", "has no assertion, status 200"),
    (200, '["x"]', "failed", "has no assertion, status 200"),
])
def test_the_login_servers_answer_is_read_and_told_without_the_assertion(status, text, kind, words):
    assertion, got, line = ladder.read_login_answer(status, text)
    assert got == kind
    if kind is None:
        assert assertion == ASSERTION and line is None
    else:
        assert assertion is None and words in line and "\n" not in line and ASSERTION not in line


def test_reasons_and_popups_become_one_plain_line():
    assert ladder.one_line("You are <b>locked</b>||Appeal at &lt;help&gt;") == "You are locked / Appeal at <help>"
    assert len(ladder.one_line("x" * 1000, 50)) == 50
    client = SimpleNamespace(account_configuration=SimpleNamespace(password=PASSWORD))
    assert ladder.masked(f"a {PASSWORD} b", client) == "a *** b"


# =============================================================================================
# A fake login server and a fake Showdown
# =============================================================================================

class FakeLoginServer:
    """act=login answered as asked: good, wrong, false, html, timeout. Keeps only whether the fields were right."""

    def __init__(self, mode: str):
        self.mode = mode
        self.requests = []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                from urllib.parse import parse_qs

                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
                form = {k: v[0] for k, v in parse_qs(body).items()}
                fake.requests.append({"act": form.get("act"), "name": form.get("name"),
                                      "password_right": form.get("pass") == PASSWORD, "challstr": bool(form.get("challstr"))})
                status, text = 200, ""
                if fake.mode == "good":
                    text = ']' + json.dumps({"actionsuccess": True, "assertion": ASSERTION, "curuser": {"loggedin": True}})
                elif fake.mode == "wrong":
                    text = ']{"actionsuccess":false,"assertion":";;Wrong password."}'
                elif fake.mode == "false":
                    text = ']{"actionsuccess":false,"assertion":false}'
                elif fake.mode == "html":
                    status = 403
                    text = "<!DOCTYPE html><html><head><title>Attention Required! | Cloudflare</title></head><body>Sorry</body></html>"
                elif fake.mode == "timeout":
                    time.sleep(3.0)
                    text = ']' + json.dumps({"actionsuccess": True, "assertion": ASSERTION})
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(text.encode())
                except OSError:
                    pass

            def log_message(self, *a):
                pass

        self.port = free_port()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/action.php?"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class FakeShowdown:
    """
    A websocket server that speaks the login part of Showdown's protocol: the guest's |updateuser|
    and |challstr| on connecting; after /trn, the lines of `before`, then after `delay` s the
    answer: confirm (|updateuser| with the name), locked (the same with the lock mark), nametaken,
    or silent (nothing).
    """

    def __init__(self, mode: str, delay: float = 0.0, before=()):
        from websockets.asyncio.server import serve

        self.mode, self.delay, self.before = mode, delay, list(before)
        self.received = []
        self.port = free_port()
        self.loop = asyncio.new_event_loop()
        started = threading.Event()

        async def handler(ws):
            await ws.send("|updateuser| Guest 4242|0|1|{}")
            await ws.send("|challstr|" + CHALLSTR)
            async for msg in ws:
                self.received.append(msg)
                if msg.startswith("|/trn "):
                    asyncio.ensure_future(self.answer(ws, msg[len("|/trn "):].split(",", 1)[0]))

        async def main():
            self.server = await serve(handler, "127.0.0.1", self.port)
            started.set()

        def run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(main())
            self.loop.run_forever()

        threading.Thread(target=run, daemon=True).start()
        assert started.wait(10)

    async def answer(self, ws, name):
        try:
            for line in self.before:
                await ws.send(line.format(name=name))
            await asyncio.sleep(self.delay)
            if self.mode == "confirm":
                await ws.send(f"|updateuser| {name}|1|1|{{}}")
            elif self.mode == "locked":
                await ws.send(f"|updateuser|\u203d{name}|1|1|{{}}")
            elif self.mode == "nametaken":
                await ws.send(f"|nametaken|{name}|Your authentication token was invalid.")
        except Exception:
            pass

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/showdown/websocket"

    def trn(self):
        return [m for m in self.received if m.startswith("|/trn ")]

    def close(self) -> None:
        async def stop():
            self.server.close()
            await self.server.wait_closed()

        try:
            asyncio.run_coroutine_threadsafe(stop(), self.loop).result(timeout=10)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)


def run_login(tmp_path, monkeypatch, capsys, login="good", showdown="confirm", delay=0.0, before=(), password=PASSWORD,
              login_wait=30.0):
    """One first login, as ladder.py does it, through poke-env's real PSClient; what it printed and wrote."""
    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.concurrency import POKE_LOOP
    from poke_env.ps_client import ps_client as psc

    log = ladder.ConsoleLog(str(tmp_path / "log"), secret=None)     # no mask for the password: a leak would show
    monkeypatch.setattr(sys, "stdout", log.wrap(sys.stdout, "out"))
    monkeypatch.setattr(sys, "stderr", log.wrap(sys.stderr, "err"))
    http_fake, ws_fake = FakeLoginServer(login), FakeShowdown(showdown, delay, before)
    name = f"logintest{next(NAMES)}"
    exit_file = str(tmp_path / "last_exit.txt")
    client = psc.PSClient(AccountConfiguration(name, password), log_level=40, start_listening=False,
                          server_configuration=ServerConfiguration(ws_fake.url, http_fake.url))
    try:
        ladder.watch_messages(client, SimpleNamespace())
        client._listening_coroutine = asyncio.run_coroutine_threadsafe(client.listen(), client.loop)
        args = SimpleNamespace(login_timeout=20, login_wait=login_wait, exit_file=exit_file)
        began = time.monotonic()
        code = asyncio.run_coroutine_threadsafe(ladder.first_login(client, args, name), POKE_LOOP).result(timeout=90)
        took = time.monotonic() - began
        time.sleep(0.3)
    finally:
        try:
            asyncio.run_coroutine_threadsafe(client.websocket.close(), client.loop).result(timeout=5)
        except Exception:
            pass
        ws_fake.close()
        http_fake.close()
    out = capsys.readouterr()
    text = out.out + out.err
    logged = "".join(open(p, encoding="utf-8").read() for p in glob.glob(str(tmp_path / "log" / "*.log")))
    noted = open(exit_file, encoding="utf-8").read() if os.path.exists(exit_file) else ""
    for where in (text, logged, noted):
        assert PASSWORD not in where and ASSERTION not in where and "ASSERT-SECRET" not in where
        assert "Traceback" not in where and CHALLSTR.split("|")[1] not in where
    return SimpleNamespace(code=code, out=text, log=logged, note=noted, took=took, client=client, name=name,
                           http=http_fake, ws=ws_fake)


# =============================================================================================
# The login server's answers
# =============================================================================================

def test_a_good_assertion_is_sent_to_showdown_and_the_confirmation_logs_in(tmp_path, monkeypatch, capsys):
    r = run_login(tmp_path, monkeypatch, capsys)
    assert r.code is None and r.client.logged_in.is_set() and r.note == ""
    assert r.http.requests == [{"act": "login", "name": r.name, "password_right": True, "challstr": True}]
    assert r.ws.trn() == [f"|/trn {r.name},0,{ASSERTION}"]          # the assertion went, and only there
    assert "the login server accepted it (status 200), name %s sent to Showdown" % r.name in r.out
    assert "logged in as %s: Showdown confirmed it" % r.name in r.out and "LOGIN FAILED" not in r.out
    assert "logged in as" in r.log                                   # the console log has the same lines


@pytest.mark.parametrize("mode, reason", [("wrong", "Wrong password."), ("false", "no reason given (actionsuccess false)")])
def test_a_refused_password_is_one_line_exit_4_and_nothing_sent_or_retried(mode, reason, tmp_path, monkeypatch, capsys):
    r = run_login(tmp_path, monkeypatch, capsys, login=mode)
    assert r.code == 4 and not r.client.logged_in.is_set() and r.ws.trn() == []
    assert len(r.http.requests) == 1                                 # asked once, never again
    line = f"login failed: the login server refused the login (status 200): {reason}"
    assert r.out.count(line) == 1
    assert "LOGIN FAILED for %s: the login server refused the login (status 200): %s" % (r.name, reason) in r.out
    assert "type the password carefully." in r.out and r.took < 10
    assert r.note == f"password: the login server refused the login (status 200): {reason}\n"


def test_a_web_page_in_place_of_the_json_is_told_as_a_likely_block(tmp_path, monkeypatch, capsys):
    r = run_login(tmp_path, monkeypatch, capsys, login="html")
    assert r.code == 2 and r.ws.trn() == []
    assert "the login server answered with a web page, status 403 (likely a block on this address)" in r.out
    assert "Cloudflare" not in r.out and "type the password" not in r.out
    assert r.note.startswith("login: the login server answered with a web page, status 403")


def test_no_answer_from_the_login_server_in_time_is_one_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ladder, "LOGIN_REQUEST_TIMEOUT", 1.0)
    r = run_login(tmp_path, monkeypatch, capsys, login="timeout")
    assert r.code == 2 and r.ws.trn() == [] and r.took < 8
    assert r.out.count("login request failed: ReadTimeout") == 2       # the failure, and the run's last line
    assert "LOGIN FAILED" in r.out and r.note.startswith("login: login request failed: ReadTimeout")


# =============================================================================================
# Showdown's answers
# =============================================================================================

def test_a_confirmation_12_s_late_is_waited_for_with_a_line_every_5_s(tmp_path, monkeypatch, capsys):
    r = run_login(tmp_path, monkeypatch, capsys, delay=12.0)
    assert r.code is None and r.client.logged_in.is_set() and 11.5 < r.took < 20
    assert "waiting for Showdown to confirm the login (5 s)" in r.out and "waiting for Showdown to confirm the login (10 s)" in r.out
    assert "(15 s)" not in r.out and "logged in as %s: Showdown confirmed it 12." % r.name in r.out


def test_no_confirmation_ends_after_login_wait_with_the_last_name_seen(tmp_path, monkeypatch, capsys):
    r = run_login(tmp_path, monkeypatch, capsys, showdown="silent", login_wait=6.0)
    assert r.code == 2 and 5.5 < r.took < 12
    assert "waiting for Showdown to confirm the login (5 s)" in r.out
    assert "Showdown did not confirm the login within 6 s (the last name it gave this connection: Guest 4242)" in r.out
    assert r.note.startswith("login: Showdown did not confirm the login within 6 s")


def test_nametaken_is_printed_and_ends_the_first_login_with_exit_4(tmp_path, monkeypatch, capsys):
    r = run_login(tmp_path, monkeypatch, capsys, showdown="nametaken")
    assert r.code == 4 and not r.client.logged_in.is_set() and r.took < 10
    line = f"Showdown refused the name {r.name} (nametaken): Your authentication token was invalid."
    assert r.out.count(line) == 2                                    # when it came, and the run's last line
    assert "ShowdownException" not in r.out and "type the password" not in r.out
    assert r.note.startswith("login: Showdown refused the name")


def test_a_lock_popup_is_printed_and_the_locked_name_still_logs_in(tmp_path, monkeypatch, capsys):
    popup = "|popup|You are locked due to your proxy / VPN and can't talk in chat.||Appeal: <a href='x'>help</a>"
    raw = "|raw|<div class='broadcast-red'>Your IP is in a datacenter range (#hostfilter)</div>"
    news = "|raw|<div class='infobox'>Today's tournament starts at 18:00</div>"
    r = run_login(tmp_path, monkeypatch, capsys, showdown="locked", before=[popup, raw, news])
    assert r.code is None and r.client.logged_in.is_set()
    assert "Showdown says (popup): You are locked due to your proxy / VPN and can't talk in chat. / Appeal: help" in r.out
    assert "Showdown says during the login: Your IP is in a datacenter range (#hostfilter)" in r.out
    assert "tournament" not in r.out                                 # a raw line about nothing of the login
    assert "logged in as %s" % r.name in r.out and "marks %s as locked on this connection" % r.name in r.out


# =============================================================================================
# No second try with a refused password; the reconnect's login
# =============================================================================================

class FakeWebsocket:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)


def fake_client(password=PASSWORD):
    return SimpleNamespace(account_configuration=SimpleNamespace(username="logintest", password=password),
                           server_configuration=SimpleNamespace(authentication_url="http://127.0.0.1:10999/action.php?"),
                           username="logintest", websocket=FakeWebsocket(), logged_in=asyncio.Event(),
                           logger=SimpleNamespace(critical=lambda *a: None))


def test_after_a_refusal_no_login_request_is_made_again(monkeypatch, capsys):
    asked = []

    def post(url, data, timeout):
        asked.append(url)
        return 200, ']{"actionsuccess":false,"assertion":";;Wrong password."}'

    monkeypatch.setattr(ladder, "_post_login", post)
    client = fake_client()
    for _ in range(3):                                             # three connections, three challenges
        asyncio.run(ladder.log_in_checked(client, "|challstr|" + CHALLSTR))
    assert len(asked) == 1 and client.websocket.sent == []
    out = capsys.readouterr().out
    assert out.count("login failed: the login server refused the login (status 200): Wrong password.") == 1
    assert out.count("login not tried again") == 2 and PASSWORD not in out
    state = ladder.login_state(client)
    assert state.refused and state.refused_kind == "password"


def test_a_reconnect_login_that_fails_says_it_is_tried_again(monkeypatch, capsys):
    def post(url, data, timeout):
        return 503, "<html><body>Service Unavailable</body></html>"

    monkeypatch.setattr(ladder, "_post_login", post)
    client = fake_client()
    ladder.login_state(client).ever_confirmed = True              # logged in before: this is a reconnect
    asyncio.run(ladder.log_in_checked(client, "|challstr|" + CHALLSTR))
    out = capsys.readouterr().out
    assert "web page, status 503 (likely a block on this address) (the login is tried again on the next connection)" in out
    assert not ladder.login_state(client).refused


def test_raw_lines_are_told_only_during_the_login_and_only_about_it(capsys):
    client = fake_client(password=None)
    state = ladder.login_state(client)
    state.watch_until = time.time() + 60
    assert ladder.note_server_word(client, "|raw|<b>Your name is locked</b>") is False
    assert ladder.note_server_word(client, "|raw|<b>Welcome to the server</b>") is False
    state.watch_until = time.time() - 1
    ladder.note_server_word(client, "|raw|a lock long after the login")
    out = capsys.readouterr().out
    assert "Showdown says during the login: Your name is locked" in out and "Welcome" not in out and "long after" not in out
    words = ladder.LOGIN_WORDS_RE
    assert all(words.search(t) for t in ("You are locked", "a proxy / VPN", "datacenter range", "#hostfilter", "Wrong password",
                                         "This name is taken", "username"))
    assert not any(words.search(t) for t in ("Today's tournament", "blockchain", "namespace", "approxy"))


def test_updateuser_marks_and_busy_status_are_read():
    client = fake_client(password=None)
    for ident, ok in ((" logintest", True), (" logintest@!", True), ("\u203dLoginTest", True), ("+logintest", True),
                      (" Guest 77", False), (" logintest2", False)):
        client.logged_in = asyncio.Event()
        ladder.login_state(client).confirmed_at = None
        ladder.note_updateuser(client, f"|updateuser|{ident}|1|1|{{}}")
        assert client.logged_in.is_set() is ok, ident
    client.logged_in = asyncio.Event()
    ladder.note_updateuser(client, "|updateuser| logintest|0|1|{}")   # not named yet
    assert not client.logged_in.is_set()


# =============================================================================================
# run_box_ab.sh's line when ladder.py ends
# =============================================================================================

def exit_line(tmp_path, code, note):
    with open(os.path.join(REPO, "run_box_ab.sh"), encoding="utf-8") as fh:
        text = fh.read()
    func = re.search(r"^exit_line\(\) \{.*?^\}$", text, re.S | re.M).group(0)
    exitf = tmp_path / "last_exit.txt"
    exitf.write_text(note, encoding="utf-8")
    done = subprocess.run(["bash", "-c", f"{func}\nEXITF='{exitf}'\nexit_line {code}"], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.mark.skipif(os.name == "nt", reason="bash on the box")
def test_the_box_scripts_last_line_says_why_and_what_to_do(tmp_path):
    out = exit_line(tmp_path, 4, "password: the login server refused the login (status 200): Wrong password.\n")
    assert out == ("ladder.py ended (exit 4): the login server refused the login (status 200): Wrong password. Press Enter, then "
                   "run bash cloud/box_ladder.sh start again on the laptop and type the password carefully.\n")
    out = exit_line(tmp_path, 2, "login: the login server answered with a web page, status 403 (likely a block on this address)\n")
    assert "status 403 (likely a block on this address). Press Enter, then run bash cloud/box_ladder.sh start again" in out
    assert "password" not in out and out.count("\n") == 1
    out = exit_line(tmp_path, 2, "")                       # no note (an older ladder.py): the code says it
    assert "the login did not go through" in out and "start again" in out
    out = exit_line(tmp_path, 0, "done: 3 wins, 2 losses, 5 of up to 5 games started\n")
    assert out == "ladder.py ended (exit 0): 3 wins, 2 losses, 5 of up to 5 games started.\n"
    out = exit_line(tmp_path, 4, "connection: no connection to the server within 300 s of losing it; 2 games abandoned\n")
    assert "start again" not in out and "2 games abandoned" in out
    assert "Ctrl+C" in exit_line(tmp_path, 130, "")
    out = exit_line(tmp_path, 130, "stopped: it was stopped with Ctrl+C\n")
    assert out == "ladder.py ended (exit 130): it was stopped with Ctrl+C.\n"


@pytest.mark.parametrize("where", ["password prompt", "run"])
def test_ctrl_c_is_one_line_exit_130_and_the_exit_file_says_so(where, tmp_path, monkeypatch, capsys):
    ckpt = tmp_path / "learner_update_000001.pt"
    ckpt.write_bytes(b"x")
    exitf = tmp_path / "last_exit.txt"
    for name in ("OUT", "GRAPH", "BATTLES"):
        monkeypatch.setattr(ladder, name, getattr(ladder, name))
    monkeypatch.chdir(tmp_path)                       # main() changes to the repo; put back afterwards
    monkeypatch.setattr(ladder, "take_account_lock", lambda *a, **k: None)

    def interrupted(*a, **k):
        raise KeyboardInterrupt()

    async def run(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(ladder.getpass, "getpass", interrupted)
    monkeypatch.setattr(ladder, "run", run)
    argv = ["--user", "ctrlcbot", "--checkpoint", str(ckpt), "--out", str(tmp_path / "results.jsonl"),
            "--exit-file", str(exitf), "--no-console-log"]
    argv += ["--server", "public"] if where == "password prompt" else ["--server", "local:10999"]
    monkeypatch.delenv("PS_PASSWORD", raising=False)
    assert ladder.main(argv) == 130
    out = capsys.readouterr()
    lines = [l for l in (out.out + out.err).splitlines() if l.strip()]
    assert len(lines) == 1 and "stopped with Ctrl+C" in lines[0] and "Traceback" not in out.err
    assert exitf.read_text(encoding="utf-8") == "stopped: it was stopped with Ctrl+C\n"


# =============================================================================================
# make_player: the connection only once the login handler is in place
# =============================================================================================

def find_checkpoint():
    env = os.environ.get("PSPPO_CKPT_024381")
    home = os.path.expanduser("~")
    candidates = ([env] if env else []) + [os.path.join(home, "dev", "ab-ladder-ckpt", "learner_update_024381.pt"),
                                           r"C:\Users\you\ps-ppo\runs\vm\checkpoints\learner_update_024381.pt"]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def test_make_player_connects_only_after_the_login_handler_is_in_place(monkeypatch):
    path = find_checkpoint()
    if path is None:
        pytest.skip("learner_update_024381.pt is not here")
    import poke_env.ps_client as ps_client

    seen = []

    async def listen(self):
        seen.append(getattr(self._handle_message, "__qualname__", ""))
        return None

    monkeypatch.setattr(ps_client.PSClient, "listen", listen)
    args = ladder.parse_args(["--server", "local:10999", "--checkpoint", path, "--device", "cpu", "--no-save-battles"])
    player = ladder.make_player(args, "logintest", None)
    player.ps_client._listening_coroutine.result(timeout=10)
    assert seen == ["watch_messages.<locals>.handle_message"]      # the first message would already go to ladder.py's login
