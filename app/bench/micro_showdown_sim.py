#!/usr/bin/env python
"""
Micro-benchmark of the Showdown simulator, without a Showdown server.

It runs N gen9randombattle games inside ONE node process, straight on the built simulator
(<ps-dir>/dist/sim), with a simple random-legal-move player on both sides, and reports:

  * simulator CPU per battle, per choice and per turn (this is what the server's single
    "sim" child process does)
  * team generation alone (Teams.generate) against the whole battle
  * how many requests leave the player exactly one legal action
  * the cost of the per-choice DEBUG message the simulator process sends its parent when
    config.debugsimprocesses is true (it is true in config-example.js), measured as a real
    node IPC round between a parent and a forked child
  * optionally, the two players' message streams of the first R battles, in the form a client
    receives them, for bench/micro_obs.py to replay through poke-env

Nothing here opens a port or starts a server.

    python bench/micro_showdown_sim.py --ps-dir C:/Users/you/showdown-bot/server --battles 300 \
        --record 150 --out <dir>

Writes <out>/sim_bench.json and, with --record, <out>/streams.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

JS = r"""
'use strict';
const path = require('path');
const fs = require('fs');

function hr() { return Number(process.hrtime.bigint()) / 1e6; }  // ms

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function legalAndChoice(req, rnd) {
  // Returns {n: number of legal actions in the bot's 14-action space, choice: string}
  if (req.wait) return null;
  const mons = req.side.pokemon;
  if (req.forceSwitch) {
    const reviving = mons.some(p => p.reviving);
    const can = [];
    for (let j = 1; j <= mons.length; j++) {
      const p = mons[j - 1];
      const fainted = p.condition.endsWith(' fnt');
      if (p.active && !reviving) continue;
      if (reviving ? fainted : !fainted) can.push(j);
    }
    if (!can.length) return {n: 1, choice: 'default', kind: 'switch'};
    return {n: can.length, choice: 'switch ' + can[Math.floor(rnd() * can.length)], kind: 'switch'};
  }
  if (req.active) {
    const a = req.active[0];
    const moves = [];
    a.moves.forEach((m, i) => { if (!m.disabled) moves.push(i + 1); });
    const canTera = !!a.canTerastallize;
    const bench = [];
    for (let j = 1; j <= mons.length; j++) {
      const p = mons[j - 1];
      if (!p.active && !p.condition.endsWith(' fnt')) bench.push(j);
    }
    const switches = a.trapped ? [] : bench;
    const n = moves.length * (canTera ? 2 : 1) + switches.length;
    let choice;
    // Mostly attack; switch one time in eight when it is certainly allowed.
    if (switches.length && !a.maybeTrapped && rnd() < 0.125) {
      choice = 'switch ' + switches[Math.floor(rnd() * switches.length)];
    } else if (moves.length) {
      choice = 'move ' + moves[Math.floor(rnd() * moves.length)];
      if (canTera && rnd() < 0.1) choice += ' terastallize';
    } else {
      choice = 'default';
    }
    return {n: Math.max(n, 1), choice, kind: 'move'};
  }
  return null;
}

function runSim(psDir, nBattles, nRecord, outFile, seed) {
  const sim = require(path.join(psDir, 'dist', 'sim', 'index.js'));
  const {extractChannelMessages} = require(path.join(psDir, 'dist', 'sim', 'battle.js'));
  const {BattleStream, Teams} = sim;
  const rnd = mulberry32(seed);
  const out = nRecord > 0 ? fs.openSync(outFile, 'w') : null;

  // Warm the dex and the JIT with a few throwaway battles first.
  const stats = {
    battles: 0, turns: 0, choices: 0, requests: 0, one_legal: 0, one_legal_move: 0, one_legal_switch: 0,
    legal_hist: {}, ms_start: 0, ms_players: 0, ms_choices: 0, ms_total_wall: 0,
    update_chunks: 0, update_bytes: 0, request_bytes: 0, errors: 0, inputlog_bytes_sum: 0,
  };
  const warm = 15;
  const t_all = hr();
  for (let b = -warm; b < nBattles; b++) {
    const counting = b >= 0;
    const recording = counting && b < nRecord;
    const tag = 'battle-gen9randombattle-' + (100000 + b);
    const stream = new BattleStream({keepAlive: true});
    const pending = [];   // chunks pushed by the simulator, in order
    stream.push = function (chunk) { pending.push(chunk); };
    const reqs = {p1: null, p2: null};
    let ended = false;
    const first = {p1: true, p2: true};
    let inputLogBytes = 0;

    const emit = (side, body) => {
      if (!recording) return;
      let msg = '>' + tag + '\n';
      if (first[side]) { msg += '|init|battle\n|title|benchpa vs. benchpb\n'; first[side] = false; }
      msg += body;
      fs.writeSync(out, JSON.stringify({b, p: side, m: msg}) + '\n');
    };

    const drain = () => {
      while (pending.length) {
        const chunk = pending.shift();
        const nl = chunk.indexOf('\n');
        const type = nl < 0 ? chunk : chunk.slice(0, nl);
        const data = nl < 0 ? '' : chunk.slice(nl + 1);
        if (type === 'update') {
          if (counting) { stats.update_chunks++; stats.update_bytes += data.length; }
          if (recording) {
            const ch = extractChannelMessages(data, [1, 2]);
            emit('p1', ch[1].join('\n'));
            emit('p2', ch[2].join('\n'));
          }
          // '|tie' alone would also match the '|tier|' line of every battle's header.
          if (data.indexOf('|win|') >= 0 || ('\n' + data + '\n').indexOf('\n|tie\n') >= 0) ended = true;
          if (counting) {
            let idx = -1;
            while ((idx = data.indexOf('|turn|', idx + 1)) >= 0) stats.turns++;
          }
        } else if (type === 'sideupdate') {
          const nl2 = data.indexOf('\n');
          const side = data.slice(0, nl2);
          const body = data.slice(nl2 + 1);
          if (body.startsWith('|request|')) {
            reqs[side] = JSON.parse(body.slice(9));
            if (counting) stats.request_bytes += body.length;
            emit(side, body);
          } else if (body.startsWith('|error|')) {
            if (counting) stats.errors++;
            reqs[side] = {error: true};
          }
        } else if (type === 'end') {
          ended = true;
        }
      }
    };

    let t0 = hr();
    stream._write('>start ' + JSON.stringify({formatid: 'gen9randombattle'}));
    let t1 = hr();
    stream._write('>player p1 ' + JSON.stringify({name: 'benchpa'}));
    stream._write('>player p2 ' + JSON.stringify({name: 'benchpb'}));
    let t2 = hr();
    if (counting) { stats.ms_start += t1 - t0; stats.ms_players += t2 - t1; }
    drain();

    let guard = 0;
    while (!ended && guard++ < 5000) {
      let acted = false;
      for (const side of ['p1', 'p2']) {
        const req = reqs[side];
        if (!req) continue;
        reqs[side] = null;
        let choice;
        if (req.error) {
          choice = 'default';
        } else {
          const r = legalAndChoice(req, rnd);
          if (!r) continue;
          choice = r.choice;
          if (counting) {
            stats.requests++;
            stats.legal_hist[r.n] = (stats.legal_hist[r.n] || 0) + 1;
            if (r.n === 1) { stats.one_legal++; if (r.kind === 'move') stats.one_legal_move++; else stats.one_legal_switch++; }
          }
        }
        const line = '>' + side + ' ' + choice;
        inputLogBytes += line.length + 1;
        const a = hr();
        stream._write(line);
        const bms = hr() - a;
        if (counting) { stats.ms_choices += bms; stats.choices++; stats.inputlog_bytes_sum += 400 + inputLogBytes; }
        acted = true;
        drain();
        if (ended) break;
      }
      if (!acted) break;
    }
    if (counting) stats.battles++;
  }
  stats.ms_total_wall = hr() - t_all;
  if (out !== null) fs.closeSync(out);

  // Team generation alone.
  const gen = Teams.getGenerator('gen9randombattle');
  for (let i = 0; i < 50; i++) gen.getTeam({});
  const nTeams = 600;
  const tg0 = hr();
  for (let i = 0; i < nTeams; i++) gen.getTeam({});
  stats.ms_per_team = (hr() - tg0) / nTeams;
  stats.node = process.version;
  console.log(JSON.stringify(stats));
}

function runIpcParent(n, bytes) {
  const {fork} = require('child_process');
  const child = fork(__filename, ['ipc-child', String(n), String(bytes)]);
  let got = 0;
  let debug = '';
  const c0 = process.cpuUsage();
  const t0 = hr();
  child.on('message', (m) => {
    if (typeof m === 'string') {
      if (m.startsWith('DEBUG\n')) { debug = m.slice(6); got++; return; }
      if (m.startsWith('DONE')) {
        const cu = process.cpuUsage(c0);
        const childCpu = JSON.parse(m.slice(4));
        console.log(JSON.stringify({
          n: got, bytes, wall_ms: hr() - t0,
          parent_cpu_us_per_msg: (cu.user + cu.system) / got,
          child_cpu_us_per_msg: (childCpu.user + childCpu.system) / got,
          last_len: debug.length,
        }));
        child.disconnect();
      }
    }
  });
}

function runIpcChild(n, bytes) {
  const payload = 'DEBUG\n' + 'x'.repeat(bytes);
  const c0 = process.cpuUsage();
  let i = 0;
  const step = () => {
    // Send in slices so the pipe drains, the way the simulator sends one per choice.
    for (let k = 0; k < 200 && i < n; k++, i++) process.send(payload);
    if (i < n) setImmediate(step);
    else process.send('DONE' + JSON.stringify(process.cpuUsage(c0)));
  };
  step();
}

const mode = process.argv[2];
if (mode === 'sim') {
  runSim(process.argv[3], parseInt(process.argv[4], 10), parseInt(process.argv[5], 10), process.argv[6], parseInt(process.argv[7], 10));
} else if (mode === 'ipc') {
  runIpcParent(parseInt(process.argv[3], 10), parseInt(process.argv[4], 10));
} else if (mode === 'ipc-child') {
  runIpcChild(parseInt(process.argv[3], 10), parseInt(process.argv[4], 10));
}
"""


def default_ps_dir() -> str:
    for cand in (os.environ.get("PS_DIR"), "C:/Users/you/showdown-bot/server", "/home/ubuntu/pokemon-showdown"):
        if cand and os.path.isdir(os.path.join(cand, "dist", "sim")):
            return cand
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps-dir", default=default_ps_dir(), help="built Pokemon Showdown checkout (needs dist/sim)")
    ap.add_argument("--battles", type=int, default=300)
    ap.add_argument("--record", type=int, default=0, help="write the players' message streams of the first R battles")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "ps_ppo_bench"))
    ap.add_argument("--ipc-messages", type=int, default=60000, help="0 skips the IPC test")
    args = ap.parse_args()

    node = shutil.which("node")
    if not node:
        print("node was not found on PATH")
        return 2
    if not args.ps_dir or not os.path.isdir(os.path.join(args.ps_dir, "dist", "sim")):
        print("give --ps-dir: a Pokemon Showdown checkout that has been built (dist/sim missing)")
        return 2
    os.makedirs(args.out, exist_ok=True)
    js_path = os.path.join(args.out, "micro_showdown_sim.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(JS)
    streams = os.path.join(args.out, "streams.jsonl")

    t0 = time.time()
    r = subprocess.run([node, js_path, "sim", os.path.abspath(args.ps_dir), str(args.battles), str(args.record), streams, str(args.seed)],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print("node failed:\n" + r.stderr[-3000:])
        return 1
    s = json.loads(r.stdout.strip().splitlines()[-1])
    wall = time.time() - t0

    b = max(s["battles"], 1)
    ch = max(s["choices"], 1)
    sim_ms = s["ms_start"] + s["ms_players"] + s["ms_choices"]
    res = {
        "node": s["node"], "battles": s["battles"], "wall_seconds": round(wall, 2),
        "turns_per_battle": round(s["turns"] / b, 2),
        "choices_per_battle": round(s["choices"] / b, 2),
        "requests_with_a_decision": s["requests"],
        "sim_ms_per_battle": round(sim_ms / b, 3),
        "sim_ms_start_per_battle": round(s["ms_start"] / b, 3),
        "sim_ms_two_teams_and_first_turn_per_battle": round(s["ms_players"] / b, 3),
        "sim_ms_all_choices_per_battle": round(s["ms_choices"] / b, 3),
        "sim_ms_per_choice": round(s["ms_choices"] / ch, 4),
        "sim_ms_per_decision_including_setup": round(sim_ms / ch, 4),
        "team_generation_ms_per_team": round(s["ms_per_team"], 4),
        "team_generation_share_of_battle": round(2 * s["ms_per_team"] / (sim_ms / b), 4),
        "one_legal_action_share": round(s["one_legal"] / max(s["requests"], 1), 4),
        "one_legal_action_share_move_requests": round(s["one_legal_move"] / max(s["requests"], 1), 4),
        "one_legal_action_share_forced_switches": round(s["one_legal_switch"] / max(s["requests"], 1), 4),
        "legal_action_histogram": {k: s["legal_hist"][k] for k in sorted(s["legal_hist"], key=int)},
        "update_chunks_per_battle": round(s["update_chunks"] / b, 2),
        "update_bytes_per_battle": round(s["update_bytes"] / b, 1),
        "request_bytes_per_request": round(s["request_bytes"] / max(s["requests"], 1), 1),
        "choice_errors": s["errors"],
        "debug_message_mean_bytes": round(s["inputlog_bytes_sum"] / ch, 1),
        "sim_choices_per_second_one_process": round(1000.0 / (sim_ms / ch), 1),
        "streams_file": streams if args.record > 0 else None,
    }

    if args.ipc_messages > 0:
        size = int(res["debug_message_mean_bytes"])
        r2 = subprocess.run([node, js_path, "ipc", str(args.ipc_messages), str(size)], capture_output=True, text=True, encoding="utf-8")
        if r2.returncode == 0 and r2.stdout.strip():
            ipc = json.loads(r2.stdout.strip().splitlines()[-1])
            res["debug_ipc"] = {
                "messages": ipc["n"], "bytes_each": ipc["bytes"],
                "sim_side_cpu_us_per_message": round(ipc["child_cpu_us_per_msg"], 2),
                "main_side_cpu_us_per_message": round(ipc["parent_cpu_us_per_msg"], 2),
                "sim_side_overhead_vs_choice": round(ipc["child_cpu_us_per_msg"] / 1000.0 / res["sim_ms_per_choice"], 4),
            }
        else:
            res["debug_ipc"] = {"error": (r2.stderr or "no output")[-500:]}

    with open(os.path.join(args.out, "sim_bench.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
