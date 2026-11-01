"""
Faults for testing ladder.py against a LOCAL Showdown server: a lost connection, a stalled one, the
search helpers going away. Nothing here talks to the public server.

    python bench/ladder_faults.py proxy --listen 9901 --target 9900 --control 9902
        a TCP proxy between the bot (ladder.py --server local:9901) and the server. Its control
        port takes one command per connection:
    python bench/ladder_faults.py cut --control 9902          every connection through the proxy is
                                                              reset at once (the bot and the server both
                                                              see it go, without a websocket close)
    python bench/ladder_faults.py stall 20 --control 9902     nothing passes either way for 20 s; the
                                                              connections stay open
    python bench/ladder_faults.py kick USER --repl DIR/logs/repl/app
        the server itself closes every connection of USER (Showdown's REPL socket: a server started
        with exports.repl, in a copy of Showdown of its own)
    python bench/ladder_faults.py helpers PID [--freeze]
        every search helper of the ladder.py with this PID is killed at once (WSL gone), or frozen
        with SIGSTOP (WSL hung); Linux only
    python bench/ladder_faults.py run --log FILE [--ladder-dir DIR] -- <ladder.py arguments>
        ladder.py (from DIR, default this checkout) with everything it puts on the wire written to
        FILE, one JSON line per message: what a run sends, to compare two versions
    python bench/ladder_faults.py report RUN_DIR [--sent FILE]
        per game: won or lost, how it ended, the least time the bot had left on its timer, whether it
        was joined again, the decisions by reason; and what was sent, by kind
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import signal
import socket
import struct
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stamp() -> str:
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------------------------
# The proxy
# ---------------------------------------------------------------------------------------------

async def proxy(listen: int, target: int, control: int) -> None:
    conns: List[Dict[str, Any]] = []
    flowing = asyncio.Event()
    flowing.set()
    counter = [0]

    async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                await flowing.wait()          # a stall holds the data here, and stops reading
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        try:
            if writer.can_write_eof():
                writer.write_eof()            # the other side's end is passed on as it came
        except Exception:
            pass

    async def handle(cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        counter[0] += 1
        n = counter[0]
        try:
            sr, sw = await asyncio.open_connection("127.0.0.1", target)
        except OSError as exc:
            print(f"{stamp()} proxy: connection {n} refused upstream: {exc}", flush=True)
            cw.close()
            return
        conn = {"n": n, "c": cw, "s": sw}
        conns.append(conn)
        print(f"{stamp()} proxy: connection {n} open", flush=True)
        await asyncio.gather(pipe(cr, sw), pipe(sr, cw))
        for w in (cw, sw):
            try:
                w.close()
            except Exception:
                pass
        if conn in conns:
            conns.remove(conn)
        print(f"{stamp()} proxy: connection {n} closed", flush=True)

    def reset(writer: asyncio.StreamWriter) -> None:
        try:
            sock = writer.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            writer.transport.abort()
        except Exception:
            pass

    async def command(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        words = (await reader.readline()).decode("utf-8", "replace").split()
        answer = "?"
        if words[:1] == ["cut"]:
            cut = list(conns)
            for conn in cut:
                reset(conn["c"])
                reset(conn["s"])
            answer = f"cut {len(cut)} connection(s)"
        elif words[:1] == ["stall"] and len(words) > 1:
            seconds = float(words[1])
            flowing.clear()
            asyncio.get_running_loop().call_later(seconds, flowing.set)
            answer = f"stalled {len(conns)} connection(s) for {seconds:.0f} s"
        elif words[:1] == ["status"]:
            answer = f"{len(conns)} connection(s), {'flowing' if flowing.is_set() else 'stalled'}"
        print(f"{stamp()} proxy: {answer}", flush=True)
        writer.write((answer + "\n").encode())
        await writer.drain()
        writer.close()

    await asyncio.start_server(handle, "127.0.0.1", listen)
    await asyncio.start_server(command, "127.0.0.1", control)
    print(f"{stamp()} proxy: 127.0.0.1:{listen} -> 127.0.0.1:{target}, control on {control}", flush=True)
    await asyncio.Event().wait()


def control(port: int, text: str) -> str:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall((text + "\n").encode())
        return s.makefile().readline().strip()


# ---------------------------------------------------------------------------------------------
# The server kicks a user, the helpers go away
# ---------------------------------------------------------------------------------------------

def kick(user: str, repl: str) -> str:
    """Showdown closes every connection of `user` itself (Connection.destroy, as on a server-side disconnect)."""
    uid = re.sub(r"[^a-z0-9]", "", user.lower())
    code = (f"(() => {{ const u = Users.get('{uid}'); if (!u) return 'no such user'; const n = u.connections.length; "
            f"for (const c of [...u.connections]) c.destroy(); return 'closed ' + n + ' connection(s) of {uid}'; }})()\n")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect(repl)
        time.sleep(0.3)
        s.sendall(code.encode())
        out = b""
        end = time.time() + 3
        while time.time() < end:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            out += chunk
            if b"closed " in out or b"no such user" in out:
                break
    text = out.decode("utf-8", "replace")
    found = re.findall(r"'(closed [^']*|no such user)'", text)
    return found[-1] if found else text.strip()[-200:]


def helper_pids(ladder_pid: int) -> List[int]:
    pids = []
    for path in glob.glob("/proc/[0-9]*/stat"):
        try:
            with open(path) as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if int(fields[1]) != ladder_pid:          # the parent
                continue
            pid = int(path.split("/")[2])
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if b"fp_bridge_server" in f.read():
                    pids.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return sorted(pids)


def helpers(ladder_pid: int, freeze: bool) -> str:
    pids = helper_pids(ladder_pid)
    sig = signal.SIGSTOP if freeze else signal.SIGKILL
    for pid in pids:                                   # all at once: each leads a process group of its own
        try:
            os.killpg(pid, sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
    return f"{'froze' if freeze else 'killed'} {len(pids)} helper(s): {pids}"


# ---------------------------------------------------------------------------------------------
# A run with what it sends written down, and the report
# ---------------------------------------------------------------------------------------------

def run_logged(log: str, ladder_dir: str, argv: List[str]) -> int:
    bench = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [ladder_dir] + [p for p in sys.path if os.path.abspath(p or ".") not in (bench, ladder_dir)]
    from poke_env.ps_client.ps_client import PSClient

    original = PSClient.send_message
    out = open(log, "a", encoding="utf-8")

    servers = ([argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--server"]
               + [a.split("=", 1)[1] for a in argv if a.startswith("--server=")])
    if not servers or not all(s.startswith("local:") for s in servers):
        raise SystemExit("ladder_faults.py run is for a local server only: give --server local:PORT")

    async def send_message(self: Any, message: str, room: str = "", message_2: Any = None):
        shown = message
        if isinstance(message, str) and message.startswith("/trn "):
            shown = message.split(",", 1)[0] + ",<login not logged>"   # never the login's assertion
        out.write(json.dumps({"t": round(time.time(), 3), "room": room, "message": shown, "message_2": message_2}) + "\n")
        out.flush()
        return await original(self, message, room, message_2)

    PSClient.send_message = send_message
    import ladder

    return ladder.main(argv)


def sent_kinds(path: str) -> Dict[str, int]:
    kinds: Dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            words = str(d["message"]).split()
            kind = words[0] if words else "?"
            if kind == "/choose" and len(words) > 1:
                kind = "/choose " + words[1] + (" +tera" if "terastallize" in words else "")
            kind += " (battle room)" if str(d.get("room") or "").startswith("battle-") else ""
            kinds[kind] = kinds.get(kind, 0) + 1
    return kinds


TIME_RE = re.compile(r"Time left: (\d+) sec this turn \| (\d+) sec total")


def report(run_dir: str, sent: str = "") -> int:
    files = sorted(glob.glob(os.path.join(run_dir, "battles", "*.jsonl")), key=os.path.getmtime)
    print(f"{'battle':<34} {'result':<6} {'turns':>5} {'ended by':<11} {'least left':>10} {'rejoined':<8} decisions by reason")
    for path in files:
        result, turns, ending, rejoined, least = "?", 0, "", False, None
        reasons: Dict[str, int] = {}
        me = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                data = d.get("data")
                if d["dir"] == "note" and isinstance(data, dict):
                    if "battle" in data and "user" in data:
                        me = data["user"]
                    if "rejoined" in data:
                        rejoined = True
                    if "decision" in data:
                        r = data["decision"].get("reason") or data["decision"].get("source")
                        reasons[r] = reasons.get(r, 0) + 1
                    if "result" in data:
                        res = data["result"]
                        result = {True: "WIN", False: "LOSS", None: "?"}[res.get("won")]
                        turns = res.get("turns", 0)
                elif d["dir"] == "in" and isinstance(data, str):
                    for m in TIME_RE.finditer(data):
                        left = int(m.group(1))
                        least = left if least is None else min(least, left)
                    if "lost due to inactivity" in data:
                        ending = "inactivity"
                    elif "forfeited." in data:
                        ending = ending or "forfeit"
                    elif "|win|" in data and not ending:
                        ending = "KO"
        name = os.path.basename(path)[:-6]
        print(f"{name:<34} {result:<6} {turns:>5} {ending:<11} {('-' if least is None else str(least) + ' s'):>10} "
              f"{'yes' if rejoined else 'no':<8} {json.dumps(reasons, sort_keys=True)}")
    if sent and os.path.isfile(sent):
        print("\nsent, by kind:")
        for kind, n in sorted(sent_kinds(sent).items()):
            print(f"  {n:6d}  {kind}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("proxy")
    p.add_argument("--listen", type=int, required=True)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--control", type=int, required=True)
    p = sub.add_parser("cut")
    p.add_argument("--control", type=int, required=True)
    p = sub.add_parser("stall")
    p.add_argument("seconds", type=float)
    p.add_argument("--control", type=int, required=True)
    p = sub.add_parser("kick")
    p.add_argument("user")
    p.add_argument("--repl", required=True)
    p = sub.add_parser("helpers")
    p.add_argument("pid", type=int)
    p.add_argument("--freeze", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--log", required=True)
    p.add_argument("--ladder-dir", default=HERE)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("report")
    p.add_argument("run_dir")
    p.add_argument("--sent", default="")
    args = ap.parse_args()
    if args.cmd == "proxy":
        asyncio.run(proxy(args.listen, args.target, args.control))
    elif args.cmd == "cut":
        print(f"{stamp()} {control(args.control, 'cut')}")
    elif args.cmd == "stall":
        print(f"{stamp()} {control(args.control, f'stall {args.seconds}')}")
    elif args.cmd == "kick":
        print(f"{stamp()} {kick(args.user, args.repl)}")
    elif args.cmd == "helpers":
        print(f"{stamp()} {helpers(args.pid, args.freeze)}")
    elif args.cmd == "run":
        rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
        return run_logged(args.log, os.path.abspath(args.ladder_dir), rest)
    elif args.cmd == "report":
        return report(args.run_dir, args.sent)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
