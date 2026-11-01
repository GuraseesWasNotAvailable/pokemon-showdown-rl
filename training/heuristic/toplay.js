#!/usr/bin/env node
// Downloads the highest-rated public replays of a format, so the bot's habits can be compared
// with what the best players do.
//
//   node heuristic/toplay.js fetch [--format gen9randombattle] [--pages 8] [--out heuristic/cache/top]
//   node heuristic/toplay.js stats [--dir heuristic/cache/top] [--min 1700] [--mine <battles folder>]
//
// `stats` prints the same numbers for the top players and, with --mine, for the bot's own
// recorded games: when Terastallization is spent, how often they switch, what they do the turn
// after the opponent sets up, how they treat a Pokemon that keeps healing, and so on.
'use strict';

const fs = require('fs');
const path = require('path');
const https = require('https');

const toID = s => ('' + (s || '')).toLowerCase().replace(/[^a-z0-9]+/g, '');

function get(url) {
	return new Promise((resolve, reject) => {
		https.get(url, {headers: {'User-Agent': 'psbot-research'}}, res => {
			if (res.statusCode !== 200) { res.resume(); return reject(new Error('HTTP ' + res.statusCode)); }
			let buf = '';
			res.setEncoding('utf8');
			res.on('data', d => { buf += d; });
			res.on('end', () => resolve(buf));
		}).on('error', reject);
	});
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function fetchAll(opts) {
	fs.mkdirSync(opts.out, {recursive: true});
	let got = 0, seen = 0;
	for (let page = 1; page <= opts.pages; page++) {
		let list;
		try {
			list = JSON.parse(await get(`https://replay.pokemonshowdown.com/search.json?format=${opts.format}&sort=rating&page=${page}`));
		} catch (e) {
			console.error('page ' + page + ': ' + e.message);
			break;
		}
		if (!list.length) break;
		for (const r of list) {
			seen++;
			const file = path.join(opts.out, r.id + '.json');
			if (fs.existsSync(file)) continue;
			try {
				const body = JSON.parse(await get(`https://replay.pokemonshowdown.com/${r.id}.json`));
				fs.writeFileSync(file, JSON.stringify({id: r.id, rating: r.rating || body.rating || 0, players: body.players, uploadtime: body.uploadtime, log: body.log}));
				got++;
			} catch (e) {
				console.error(r.id + ': ' + e.message);
			}
			await sleep(120);
		}
		console.log(`page ${page}: ${list.length} listed, ${got} new so far`);
		if (list.length < 51) break;
	}
	console.log(`${got} replays saved to ${opts.out} (${seen} listed)`);
}

const SETUP = new Set(['swordsdance', 'nastyplot', 'calmmind', 'dragondance', 'quiverdance', 'bulkup', 'shellsmash', 'bellydrum', 'agility', 'rockpolish', 'coil', 'curse', 'irondefense', 'cosmicpower', 'shiftgear', 'victorydance', 'tidyup', 'clangoroussoul', 'trailblaze', 'tailglow', 'noretreat', 'growth', 'workup', 'honeclaws', 'autotomize', 'geomancy', 'filletaway', 'acidarmor', 'amnesia']);
const HEAL = new Set(['recover', 'roost', 'slackoff', 'softboiled', 'milkdrink', 'moonlight', 'synthesis', 'morningsun', 'shoreup', 'strengthsap', 'rest', 'wish', 'healorder']);
const PHAZE = new Set(['haze', 'clearsmog', 'roar', 'whirlwind', 'dragontail', 'circlethrow', 'encore', 'taunt']);
const PIVOT = new Set(['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport', 'batonpass', 'chillyreception', 'shedtail']);

// One side's actions per turn, read from a protocol log.
function parse(lines) {
	const g = {turns: [], winner: '', players: {}, teraTurn: {}, faintsAtTera: {}};
	let cur = null, turn = 0;
	const fainted = {p1: 0, p2: 0};
	const hp = {p1: 100, p2: 100};
	const start = () => ({turn, p1: {}, p2: {}, hp0: {p1: hp.p1, p2: hp.p2}, fainted0: {p1: fainted.p1, p2: fainted.p2}});
	for (const ln of lines) {
		const p = ln.split('|');
		const k = p[1];
		if (k === 'player' && p[3]) g.players[p[2]] = p[3];
		if (k === 'turn') { turn = +p[2]; cur = start(); g.turns.push(cur); continue; }
		if (k === 'win') g.winner = p[2];
		if (!cur) continue;
		const side = (p[2] || '').slice(0, 2);
		if (side !== 'p1' && side !== 'p2') continue;
		if (k === 'move') {
			if (/\[from\]/.test(ln) && !/\[from\] ?lockedmove/.test(ln)) continue; // Sleep Talk, Dancer and the like
			if (!cur[side].move) cur[side].move = toID(p[3]);
		} else if (k === 'switch' || k === 'drag') {
			const pct = /^(\d+)\/(\d+)/.exec(p[4] || '');
			hp[side] = pct ? Math.round(100 * +pct[1] / +pct[2]) : 100;
			if (k === 'switch' && !cur[side].move && !cur[side].faintedThisTurn && !cur[side].switched) cur[side].switched = true;
			else if (k === 'switch') cur[side].replaced = true;
		} else if (k === '-terastallize') {
			cur[side].tera = p[3];
			if (g.teraTurn[side] === undefined) { g.teraTurn[side] = turn; g.faintsAtTera[side] = {mine: fainted[side], theirs: fainted[side === 'p1' ? 'p2' : 'p1']}; }
		} else if (k === 'faint') {
			fainted[side]++;
			cur[side].faintedThisTurn = true;
		} else if (k === '-damage' || k === '-heal') {
			const m = /^(\d+)\/(\d+)/.exec(p[3] || '');
			if (m) hp[side] = Math.round(100 * +m[1] / +m[2]);
			else if (/^0 fnt/.test(p[3] || '')) hp[side] = 0;
		} else if (k === 'cant') {
			cur[side].cant = true;
		}
	}
	g.totalTurns = turn;
	return g;
}

function tally() {
	return {games: 0, sideTurns: 0, switches: 0, pivots: 0, setup: 0, heal: 0, attacks: 0, other: 0,
		tera: 0, teraTurns: [], teraWithKO: 0, teraEarly: 0, teraMineDown: [], teraTheirsDown: [],
		afterOppSetup: {attack: 0, switch: 0, phaze: 0, setup: 0, heal: 0, other: 0, n: 0},
		afterOppHeal: {attack: 0, switch: 0, phaze: 0, setup: 0, heal: 0, other: 0, status: 0, n: 0},
		lead: {attack: 0, switch: 0, setup: 0, other: 0, n: 0},
		setupHp: [], healHp: [], repeatHealLoop: 0, turns: []};
}

function classify(a) {
	if (!a) return 'other';
	if (a.switched) return 'switch';
	if (!a.move) return 'other';
	if (SETUP.has(a.move)) return 'setup';
	if (HEAL.has(a.move)) return 'heal';
	if (PHAZE.has(a.move)) return 'phaze';
	return 'attack';
}

function add(T, g, side) {
	const foe = side === 'p1' ? 'p2' : 'p1';
	T.games++;
	T.turns.push(g.totalTurns);
	if (g.teraTurn[side] !== undefined) {
		T.tera++;
		T.teraTurns.push(g.teraTurn[side]);
		if (g.teraTurn[side] <= 2) T.teraEarly++;
		T.teraMineDown.push(g.faintsAtTera[side].mine);
		T.teraTheirsDown.push(g.faintsAtTera[side].theirs);
	}
	let streak = 0;
	g.turns.forEach((t, i) => {
		const a = t[side], b = t[foe];
		if (!a.move && !a.switched) return;
		T.sideTurns++;
		const c = classify(a);
		if (a.switched) T.switches++;
		else if (PIVOT.has(a.move)) T.pivots++;
		if (c === 'setup') { T.setup++; T.setupHp.push(t.hp0[side]); }
		else if (c === 'heal') { T.heal++; T.healHp.push(t.hp0[side]); }
		else if (c === 'attack') T.attacks++;
		else if (c !== 'switch') T.other++;
		if (a.tera && b.faintedThisTurn) T.teraWithKO++;
		if (i === 0) { T.lead.n++; T.lead[c === 'heal' || c === 'phaze' ? 'other' : c]++; }
		const prev = g.turns[i - 1];
		if (prev && !prev[foe].faintedThisTurn && !t[foe].replaced) {
			const pc = classify(prev[foe]);
			if (pc === 'setup' && !prev[side].faintedThisTurn) { T.afterOppSetup.n++; T.afterOppSetup[c]++; }
			if (pc === 'heal' && !prev[side].faintedThisTurn) { T.afterOppHeal.n++; T.afterOppHeal[c]++; }
		}
		// Attacking again and again into a Pokemon that heals each time.
		if (c === 'attack' && classify(b) === 'heal') streak++; else if (c !== 'attack') streak = 0;
		if (streak === 3) T.repeatHealLoop++;
	});
}

const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
const median = xs => { if (!xs.length) return 0; const s = xs.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const pct = (a, b) => b ? (100 * a / b).toFixed(1) + '%' : '-';
const share = o => Object.keys(o).filter(k => k !== 'n').map(k => `${k} ${pct(o[k], o.n)}`).join(', ');

function show(name, T) {
	console.log(`\n== ${name}: ${T.games} sides, ${T.sideTurns} turns`);
	console.log(`  game length: mean ${mean(T.turns).toFixed(1)} turns, median ${median(T.turns)}`);
	console.log(`  per turn: switch ${pct(T.switches, T.sideTurns)}, pivot move ${pct(T.pivots, T.sideTurns)}, setup ${pct(T.setup, T.sideTurns)}, heal ${pct(T.heal, T.sideTurns)}, attack ${pct(T.attacks, T.sideTurns)}`);
	console.log(`  Tera used in ${pct(T.tera, T.games)} of games; turn used: mean ${mean(T.teraTurns).toFixed(1)}, median ${median(T.teraTurns)}; by turn 2: ${pct(T.teraEarly, T.tera)}; with a KO that turn: ${pct(T.teraWithKO, T.tera)}`);
	console.log(`  when Tera is spent: own Pokemon already down ${mean(T.teraMineDown).toFixed(2)}, theirs down ${mean(T.teraTheirsDown).toFixed(2)}`);
	console.log(`  first turn: ${share(T.lead)}`);
	console.log(`  turn after the opponent set up (${T.afterOppSetup.n}): ${share(T.afterOppSetup)}`);
	console.log(`  turn after the opponent healed (${T.afterOppHeal.n}): ${share(T.afterOppHeal)}`);
	console.log(`  HP when setting up: mean ${mean(T.setupHp).toFixed(0)}%, median ${median(T.setupHp)}%; HP when healing: mean ${mean(T.healHp).toFixed(0)}%, median ${median(T.healHp)}%`);
	console.log(`  three attacks in a row into a healing Pokemon: ${T.repeatHealLoop} times (${(T.repeatHealLoop / Math.max(1, T.games)).toFixed(2)} per side)`);
}

function walk(dir) {
	const out = [];
	if (!fs.existsSync(dir)) return out;
	for (const f of fs.readdirSync(dir)) {
		const p = path.join(dir, f);
		if (fs.statSync(p).isDirectory()) out.push(...walk(p));
		else if (f.endsWith('.json')) out.push(p);
	}
	return out;
}

function stats(opts) {
	const top = tally(), topWin = tally(), topLose = tally();
	let n = 0;
	for (const f of walk(opts.dir)) {
		let r;
		try { r = JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { continue; }
		if (!r.log || (r.rating || 0) < opts.min) continue;
		const g = parse(r.log.split('\n'));
		if (!g.turns.length) continue;
		n++;
		for (const side of ['p1', 'p2']) {
			add(top, g, side);
			if (g.winner) add(toID(g.players[side]) === toID(g.winner) ? topWin : topLose, g, side);
		}
	}
	console.log(`${n} replays rated ${opts.min}+`);
	show('top players, both sides', top);
	show('top players, winning side', topWin);
	show('top players, losing side', topLose);
	if (opts.mine) {
		const all = tally(), won = tally(), lost = tally();
		for (const f of walk(opts.mine)) {
			let r;
			try { r = JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { continue; }
			if (!r.log || !r.me) continue;
			const g = parse(r.log);
			if (!g.turns.length) continue;
			add(all, g, r.me.side);
			add(r.result === 'win' ? won : lost, g, r.me.side);
		}
		show('the bot', all);
		show('the bot, games won', won);
		show('the bot, games lost', lost);
	}
}

(async () => {
	const args = process.argv.slice(2);
	const opt = (name, def) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : def; };
	const cmd = args[0];
	if (cmd === 'fetch') {
		await fetchAll({format: opt('format', 'gen9randombattle'), pages: +opt('pages', 8), out: opt('out', path.join(__dirname, 'cache', 'top'))});
	} else if (cmd === 'stats') {
		stats({dir: opt('dir', path.join(__dirname, 'cache', 'top')), min: +opt('min', 1700), mine: opt('mine', '')});
	} else {
		console.error('usage: node heuristic/toplay.js fetch|stats [options]');
		process.exit(2);
	}
})().catch(e => { console.error(e); process.exit(1); });
