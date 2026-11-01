#!/usr/bin/env node
// Self-play battle logs from both seats, for tests/test_read_reward.py (the zero-sum check).
//
//   node tests/selfplay_logs.js N out.jsonl [seed]          (PS_DIR defaults to ~/pokemon-showdown)
//
// Plays N gen9randombattle games in Showdown's own simulator (BattleStream) between two random players
// that switch a quarter of the time and Terastallize now and then, so voluntary switches and hits on
// switch-ins are common. Each output line is one game: what each seat received, in order, and what it
// chose: {"game": i, "p1": [["in", chunk] | ["choice", action, forced]], "p2": [...]}. A chunk is exactly
// what that player's stream delivered (its own side with exact HP, the other in percent), requests
// included. action: 0-3 a move slot, 10-13 the same with Tera, 4 a switch.
'use strict';

const fs = require('fs');
const path = require('path');
const PS = process.env.PS_DIR || path.join(process.env.HOME || '', 'pokemon-showdown');
const sim = require(path.join(PS, 'dist', 'sim'));
const {BattleStream, getPlayerStreams} = sim.BattleStream ? sim : require(path.join(PS, 'dist', 'sim', 'battle-stream'));

function rng(seed) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6D2B79F5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function choose(req, rnd) {
	if (req.wait || req.teamPreview) return null;
	const pick = xs => xs[Math.floor(rnd() * xs.length)];
	const bench = req.side.pokemon.map((p, i) => ({p, slot: i + 1}))
		.filter(x => !x.p.active && !x.p.condition.endsWith(' fnt'));
	if (req.forceSwitch) {
		if (!bench.length) return {choice: 'pass', action: 4, forced: true};
		return {choice: `switch ${pick(bench).slot}`, action: 4, forced: true};
	}
	if (!req.active) return null;
	const a = req.active[0];
	const moves = a.moves.map((m, i) => ({m, slot: i + 1})).filter(x => !x.m.disabled);
	if (bench.length && !a.trapped && !a.maybeTrapped && rnd() < 0.25) {
		return {choice: `switch ${pick(bench).slot}`, action: 4, forced: false};
	}
	if (!moves.length) return {choice: 'move 1', action: 0, forced: false};
	const mv = pick(moves);
	const tera = !!a.canTerastallize && rnd() < 0.1;
	return {choice: `move ${mv.slot}${tera ? ' terastallize' : ''}`, action: mv.slot - 1 + (tera ? 10 : 0), forced: false};
}

async function playGame(index, seed) {
	const stream = new BattleStream();
	const streams = getPlayerStreams(stream);
	const logs = {p1: [], p2: []};
	let ended = false;
	const send = (side, choice) => {
		if (ended) return;
		try {
			void streams[side].write(choice);
		} catch (e) {
			if (!ended) throw e;
		}
	};
	const players = ['p1', 'p2'].map(async (side, k) => {
		const rnd = rng(seed * 2 + k);
		for await (const chunk of streams[side]) {
			logs[side].push(['in', chunk]);
			for (const line of chunk.split('\n')) {
				if (line.startsWith('|win|') || line === '|tie' || line.startsWith('|tie|')) ended = true;
			}
			for (const line of chunk.split('\n')) {
				if (line.startsWith('|request|')) {
					const c = choose(JSON.parse(line.slice(9)), rnd);
					if (!c) continue;
					logs[side].push(['choice', c.action, c.forced]);
					send(side, c.choice);
				} else if (line.startsWith('|error|')) {
					process.stderr.write(`game ${index} ${side}: ${line}\n`);
					send(side, 'default');
				}
			}
			if (ended) break;
		}
	});
	void streams.omniscient.write(`>start {"formatid":"gen9randombattle"}\n>player p1 {"name":"Alice"}\n>player p2 {"name":"Bob"}`);
	await Promise.race([Promise.all(players), new Promise(r => setTimeout(r, 60000))]);
	return {game: index, p1: logs.p1, p2: logs.p2};
}

(async () => {
	const n = parseInt(process.argv[2] || '10', 10);
	const out = process.argv[3] || '/dev/stdout';
	const seed = parseInt(process.argv[4] || '1', 10);
	const lines = [];
	for (let i = 0; i < n; i++) lines.push(JSON.stringify(await playGame(i, seed * 100003 + i)));
	fs.writeFileSync(out, lines.join('\n') + '\n');
	process.exit(0);
})();
