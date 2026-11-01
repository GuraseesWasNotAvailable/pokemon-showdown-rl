#!/usr/bin/env python
"""
A stand-in for start_servers.py, for exercising pipeline.py without Node or Pokemon Showdown.

    python tests/fake_servers.py --dir IGNORED --ports 8110-8111

Same shape as the real launcher: one child process per port (here a Python process that only
accepts TCP connections on 127.0.0.1), restarted if it exits, with start_servers.py's message
"server on P exited with N, restarting". Stopping it therefore exercises the same handling of a
process tree as the real thing.

  --kill-child-after SECONDS   once, after this long, kill the first port's child (the launcher
                               then restarts it, as the real one does with a crashed server)
  --kill-child-when-file FILE  the same, as soon as FILE exists (a test creates it at the moment it
                               wants the server to go down: no race with how fast things start)
  --once-marker FILE           with either of the above: skip the kill if FILE exists, create it when killing
  --fail-first-start FILE      the first port's server exits with code 3 instead of listening, once
                               (FILE is created when it has): a server that falls over while the
                               servers are still coming up, and is started again by the launcher
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time


def parse_ports(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def serve(port: int, fail_marker: str = "") -> int:
    if fail_marker and not os.path.exists(fail_marker):
        with open(fail_marker, "w", encoding="utf-8") as f:
            f.write(f"{time.time()} port {port}\n")
        print(f"fake server on {port}: falling over before listening, this once", flush=True)
        return 3
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name != "nt":
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
    except OSError as e:
        print(f"fake server: cannot bind 127.0.0.1:{port}: {e}", flush=True)
        return 1
    s.listen(64)
    print(f"fake server listening on 127.0.0.1:{port}", flush=True)
    while True:
        try:
            conn, _ = s.accept()
            conn.close()
        except OSError:
            return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="")
    ap.add_argument("--ports", default="8110-8111")
    ap.add_argument("--heap-mb", type=int, default=0)
    ap.add_argument("--child", type=int, default=0)
    ap.add_argument("--kill-child-after", type=float, default=0.0)
    ap.add_argument("--kill-child-when-file", default=None)
    ap.add_argument("--once-marker", default=None)
    ap.add_argument("--fail-first-start", default=None)
    args = ap.parse_args()
    if args.child:
        return serve(args.child, args.fail_first_start or "")

    ports = parse_ports(args.ports)
    procs = {}
    stopping = False

    # The interpreter itself, not a venv's python.exe: on Windows that one is a small launcher
    # with the interpreter as its child, which doubles the number of processes for nothing.
    python = getattr(sys, "_base_executable", None) or sys.executable

    def start(port: int) -> None:
        extra = ["--fail-first-start", args.fail_first_start] if args.fail_first_start and port == ports[0] else []
        procs[port] = subprocess.Popen([python, os.path.abspath(__file__), "--child", str(port)] + extra)

    def stop(*_a) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    try:
        signal.signal(signal.SIGTERM, stop)
    except Exception:
        pass

    for p in ports:
        start(p)
    print(f"servers on {ports[0]}..{ports[-1]} ({len(ports)}). Ctrl+C to stop.", flush=True)
    began = time.time()
    kill_due = ((args.kill_child_after > 0 or bool(args.kill_child_when_file))
                and not (args.once_marker and os.path.exists(args.once_marker)))
    try:
        while not stopping:
            time.sleep(0.2)
            if kill_due and (os.path.exists(args.kill_child_when_file) if args.kill_child_when_file
                             else time.time() - began > args.kill_child_after):
                kill_due = False
                if args.once_marker:
                    with open(args.once_marker, "w", encoding="utf-8") as f:
                        f.write(f"{time.time()}\n")
                procs[ports[0]].kill()
                procs[ports[0]].wait()
            for p in ports:
                if procs[p].poll() is not None and not stopping:
                    print(f"server on {p} exited with {procs[p].returncode}, restarting", flush=True)
                    start(p)
    finally:
        for pr in procs.values():
            if pr.poll() is None:
                pr.terminate()
        for pr in procs.values():
            try:
                pr.wait(timeout=10)
            except Exception:
                pr.kill()
        print("servers stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
