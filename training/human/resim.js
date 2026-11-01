// Re-run Showdown inputlogs in one built Showdown version and print each player's own stream.
//
//     node human/resim.js <build dir that holds dist/> < jobs.jsonl > results.jsonl
//
// jobs.jsonl: one {"key": "...", "inputlog": "<the replay's inputlog>"} per line, all of games that
// ran on (a version equivalent to) this build. results.jsonl: one line per job,
//     {"key", "p1": [chunks], "p2": [chunks], "omniscient": [chunks], "err": null | "..."}
// where a chunk is what the simulator's player stream yields: a block of protocol lines, the
// |request| messages included, exactly the per-side view the server relays (|split| resolved).
// Used by human/rebuild.py; run on the box only (nice -n 15).
'use strict';
process.on('unhandledRejection', () => {});
process.on('uncaughtException', e => { process.stderr.write('uncaught ' + String(e).slice(0, 300) + String.fromCharCode(10)); });
const path = require('path');
const readline = require('readline');
const {BattleStream, getPlayerStreams} = require(path.resolve(process.argv[2], 'dist/sim/battle-stream'));

async function one(inputlog) {
  const input = inputlog.split('\n').filter(l => l && !l.startsWith('>version'));
  const streams = getPlayerStreams(new BattleStream());
  const out = {omniscient: [], spectator: [], p1: [], p2: []};
  const reads = Object.keys(out).map(k => (async () => { for await (const c of streams[k]) out[k].push(c); })());
  for (const l of input) streams.omniscient.write(l);
  streams.omniscient.writeEnd();
  let timer;
  const timeout = new Promise((_, rej) => { timer = setTimeout(() => rej(new Error('timeout')), 30000); });
  await Promise.race([Promise.all(reads), timeout]);
  clearTimeout(timer);
  return {p1: out.p1, p2: out.p2, omniscient: out.omniscient};
}

(async () => {
  const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
  for await (const line of rl) {
    if (!line.trim()) continue;
    let job;
    try { job = JSON.parse(line); } catch (e) { continue; }
    let res;
    try {
      res = Object.assign({key: job.key, err: null}, await one(job.inputlog));
    } catch (e) {
      res = {key: job.key, err: String(e && e.stack || e).slice(0, 500)};
    }
    process.stdout.write(JSON.stringify(res) + '\n');
  }
})();
