#!/usr/bin/env node
// Replays recorded games (the JSON files the browser bot saves) through the current bot, one
// decision at a time, and shows where it would now choose differently.
//
//   node heuristic/replay.js <game.json | folder> [--all] [--turn N] [--losses]
//
//   --all      print every decision, not only the ones that changed
//   --turn N   only turn N
//   --losses   only games that were lost
//
// The state at each decision is rebuilt from the game's own protocol log and team snapshot, so
// this needs nothing but the record. Choice locks and PP are approximated.
'use strict';

const fs = require('fs');
const path = require('path');
const {createEngine} = require('./engine');

const toID = s => ('' + (s || '')).toLowerCase().replace(/[^a-z0-9]+/g, '');
const speciesOf = details => ('' + (details || '')).split(',')[0];

function files(target) {
	const st = fs.statSync(target);
	if (!st.isDirectory()) return [target];
	const out = [];
	const walk = dir => fs.readdirSync(dir).forEach(f => {
		const p = path.join(dir, f);
		if (fs.statSync(p).isDirectory()) walk(p);
		else if (f.endsWith('.json')) out.push(p);
	});
	walk(target);
	return out.sort();
}

// The request the server would have sent for this decision.
function buildRequest(rec, d, state, movedex) {
	const side = rec.me.side;
	const mons = d.mine.map(m => {
		const team = rec.myTeam || [];
		const stem = x => ('' + x).split('-')[0];
		const base = team.find(p => speciesOf(p.details) === m.species) || team.find(p => stem(speciesOf(p.details)) === stem(m.species)) || {};
		return Object.assign({}, base, {
			condition: m.condition, active: !!m.active,
			terastallized: state.teraMon === m.species ? state.teraType : '',
		});
	});
	const req = {side: {name: rec.me.name, id: side, pokemon: mons}, rqid: state.rqid++};
	if (d.forced) {
		req.forceSwitch = [true];
		return req;
	}
	const act = mons.find(p => p.active) || mons[0];
	const locked = /^choice/.test(act.item || '') && state.lastMove && state.lastMoveMon === speciesOf(act.details) ? state.lastMove : '';
	req.active = [{
		moves: (act.moves || []).map(id => {
			const mv = movedex[id] || {};
			return {move: mv.name || id, id, pp: 8, maxpp: 8, target: mv.target || 'normal', disabled: !!locked && id !== locked};
		}),
	}];
	if (!state.teraMon && act.teraType && rec.gen === 9) req.active[0].canTerastallize = act.teraType;
	return req;
}

async function replay(file, opts, engine) {
	const rec = JSON.parse(fs.readFileSync(file, 'utf8'));
	if (!rec.log || !rec.decisions || !rec.me) return null;
	if (opts.losses && rec.result !== 'loss') return null;
	const side = rec.me.side;
	const key = 'replay|' + rec.id;
	const movedex = engine.sandbox.BattleMovedex;
	const lines = rec.log;
	const state = {rqid: 1, teraMon: '', teraType: '', lastMove: '', lastMoveMon: ''};
	let pos = 0;
	const out = {id: rec.id, result: rec.result, total: 0, changed: 0, rows: []};

	const feed = to => {
		const chunk = lines.slice(pos, to);
		// Track what the request would have reflected: my Tera, and my last move for Choice locks.
		for (const ln of chunk) {
			const p = ln.split('|');
			if (p[1] === '-terastallize' && p[2].startsWith(side)) {
				state.teraMon = (rec.myTeam || []).map(m => speciesOf(m.details)).find(s => p[2].includes(s.split('-')[0])) || p[2].replace(/^p\d[a-z]?: /, '');
				state.teraType = p[3];
			}
			if (p[1] === 'move' && p[2].startsWith(side)) { state.lastMove = toID(p[3]); state.lastMoveMon = state.activeMon; }
			if ((p[1] === 'switch' || p[1] === 'drag') && p[2].startsWith(side)) { state.activeMon = speciesOf(p[3]); state.lastMove = ''; }
		}
		if (chunk.length) engine.feed(key, rec.id, chunk.join('\n'));
		pos = to;
	};

	// Tell the engine which side is mine before any line arrives, the way a live game does, so
	// the bot's own bookkeeping (opponent switches, failed moves) follows the whole log.
	engine.feed(key, rec.id, '|request|' + JSON.stringify({wait: true, rqid: 0, side: {id: side, name: rec.me.name, pokemon: []}}));

	for (const d of rec.decisions) {
		let cut;
		if (d.forced) {
			cut = lines.findIndex((ln, i) => i >= pos && (ln.startsWith('|switch|' + side) || ln.startsWith('|drag|' + side)));
			if (cut < 0) cut = lines.length;
		} else {
			cut = lines.findIndex((ln, i) => i >= pos && ln === '|turn|' + d.turn);
			cut = cut < 0 ? pos : cut + 1;
		}
		feed(cut);
		if (opts.turn !== null && d.turn !== opts.turn) {
			engine.chose(key, d.choice);
			continue;
		}
		const req = buildRequest(rec, d, state, movedex);
		let res;
		try {
			res = await engine.decide(key, rec.id, req);
		} catch (e) {
			res = {choice: 'ERROR ' + e.message, why: '', table: []};
		}
		const same = res.choice === d.choice;
		out.total++;
		if (!same) out.changed++;
		out.rows.push({turn: d.turn, forced: d.forced, was: d.why, wasChoice: d.choice, now: res.why, nowChoice: res.choice, same, table: res.table, dbg: res.dbg});
		// Keep the bot's bookkeeping on the path the game actually took.
		engine.chose(key, d.choice);
	}
	engine.end(key);
	return out;
}

module.exports = {replay, files};

if (require.main === module) (async () => {
	const args = process.argv.slice(2);
	const target = args.find(a => !a.startsWith('--'));
	if (!target) {
		console.error('usage: node heuristic/replay.js <game.json | folder> [--all] [--turn N] [--losses]');
		process.exit(2);
	}
	const ti = args.indexOf('--turn');
	const opts = {all: args.includes('--all'), losses: args.includes('--losses'), turn: ti >= 0 ? +args[ti + 1] : null};
	const engine = createEngine({log: () => {}});
	let total = 0, changed = 0;
	for (const f of files(target)) {
		const r = await replay(f, opts, engine);
		if (!r) continue;
		total += r.total;
		changed += r.changed;
		console.log(`\n== ${r.id} (${r.result}): ${r.changed} of ${r.total} decisions differ`);
		for (const row of r.rows) {
			if (row.same && !opts.all) continue;
			const top = (row.table || []).slice(0, 5).map(o => `${o.label} ${o.v === null ? '?' : o.v.toFixed(2)}`).join(' | ');
			console.log(`  T${row.turn}${row.forced ? ' forced' : ''}: was ${row.was}${row.same ? '' : `\n      now ${row.now}`}\n      [${top}]`);
		}
	}
	console.log(`\n${changed} of ${total} decisions differ from what was played`);
	process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
