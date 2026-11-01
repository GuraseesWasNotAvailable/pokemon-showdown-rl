#!/usr/bin/env python
"""
Starts the local Pokemon Showdown servers the trainer plays on, and keeps them up.

    .venv/Scripts/python.exe start_servers.py --dir C:/Users/you/showdown-bot/server --ports 8000-8003

This is start_servers.sh for any OS: one `node pokemon-showdown start --no-security --port P`
per port, restarted if it exits, output in <dir>/logs/multi/ (or in --log-dir). Ctrl+C stops them
all. The server is built once up front and every instance starts with --skip-build: without it
each start rebuilds dist/ underneath the instances already running from the same folder.

When a server exits, one line is printed before it is started again:

    server on 8001 exited with 1, restarting

pipeline.py reads that line (it restarts the trainer, whose players on that server are gone), so
its wording is part of the interface.

Restart the servers before each training run. The spawning plugin keeps refilling battles for a
pair for a long time after the trainer that asked for them has gone.

The server needs the battle-spawning plugin from this repository. Install it once:

    copy pokemon-showdown/server/chat-plugins/rlspawn.ts into <dir>/server/chat-plugins/
    then run `node build` in <dir>

Without it the workers log in and wait forever: nothing creates the battles.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List


def parse_ports(spec: str) -> List[int]:
    out: List[int] = []
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="the Pokemon Showdown checkout (with rlspawn.ts installed and built)")
    ap.add_argument("--ports", default="8000-8009", help="e.g. 8000-8003 or 8000,8001,8005")
    ap.add_argument("--heap-mb", type=int, default=2048, help="node --max-old-space-size for each server")
    ap.add_argument("--build", action="store_true", help="run `node build` first")
    ap.add_argument("--log-dir", default=None,
                    help="where each server's own output goes, as server_<port>.out and server_<port>.err (appended to). "
                         "Default: <dir>/logs/multi. pipeline.py passes RUN/logs/servers so a run keeps its servers' output")
    args = ap.parse_args()

    ps_dir = os.path.abspath(args.dir)
    plugin_src = os.path.join(ps_dir, "server", "chat-plugins", "rlspawn.ts")
    plugin_built = os.path.join(ps_dir, "dist", "server", "chat-plugins", "rlspawn.js")
    if not os.path.isfile(plugin_src):
        print(f"rlspawn.ts is not in {os.path.dirname(plugin_src)}. Copy it there from this repository's pokemon-showdown/ folder first.", flush=True)
        return 2
    if args.build or not os.path.isfile(plugin_built):
        print("building the server...", flush=True)
        r = subprocess.run(["node", "build"], cwd=ps_dir)
        if r.returncode != 0 or not os.path.isfile(plugin_built):
            print("the build did not produce dist/server/chat-plugins/rlspawn.js", flush=True)
            return 2

    ports = parse_ports(args.ports)
    log_dir = os.path.abspath(args.log_dir) if args.log_dir else os.path.join(ps_dir, "logs", "multi")
    os.makedirs(log_dir, exist_ok=True)
    procs: Dict[int, subprocess.Popen] = {}
    stopping = False

    def start(port: int) -> None:
        # The child holds its own copies of the two files: ours are closed again at once, so a
        # server that is restarted many times does not leave a pair of open files behind each time.
        with open(os.path.join(log_dir, f"server_{port}.out"), "ab") as out, open(os.path.join(log_dir, f"server_{port}.err"), "ab") as err:
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for f in (out, err):
                f.write(f"=== {stamp} start_servers.py (pid {os.getpid()}) starts the server on port {port}\n".encode("ascii"))
                f.flush()
            procs[port] = subprocess.Popen(
                ["node", f"--max-old-space-size={args.heap_mb}", "pokemon-showdown", "start", "--no-security", "--skip-build", "--port", str(port)],
                cwd=ps_dir, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            )

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
        time.sleep(1.0)
    print(f"servers on {ports[0]}..{ports[-1]} ({len(ports)}), logs in {log_dir}. Ctrl+C to stop.", flush=True)
    try:
        while not stopping:
            time.sleep(2.0)
            for p in ports:
                if procs[p].poll() is not None and not stopping:
                    print(f"server on {p} exited with {procs[p].returncode}, restarting", flush=True)
                    start(p)
    finally:
        for p, pr in procs.items():
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
