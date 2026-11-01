#!/usr/bin/env node
// Looks inside one decision of a recorded game: every option's score, what the bot believes
// about the opponent, and how it expects the matchup to play out. For working out why the bot
// picked what it picked.
//
//   node heuristic/probe.js <game.json> <turn> [--forced] [--nth N]
'use strict';

const fs = require('fs');
const path = require('path');
const {createEngine} = require('./engine');
const {replay} = require('./replay');

(async () => {
	const args = process.argv.slice(2);
	let file = args[0];
	const turn = +args[1];
	if (!file || !turn) { console.error('usage: node heuristic/probe.js <game.json> <turn> [--forced] [--nth N]'); process.exit(2); }
	if (!fs.existsSync(file)) file = path.join(__dirname, 'lessons', file.replace(/\.json$/, '') + '.json');
	const ni = args.indexOf('--nth');
	const engine = createEngine({log: () => {}});
	const I = engine.PSBot._internals;
	// Capture the state the bot decided from.
	let captured = null;
	const origDecide = I.decide;
	const r = await replay(file, {turn, all: true, losses: false}, engine);
	const rows = r.rows.filter(x => x.turn === turn && !!x.forced === args.includes('--forced'));
	const row = rows[ni >= 0 ? +args[ni + 1] : 0];
	if (!row) { console.error('no such decision'); process.exit(1); }
	console.log(`${r.id} turn ${turn}: was "${row.was}", now "${row.now}"`);
	for (const o of row.table || []) {
		console.log(`  ${(o.v === null ? '?' : o.v.toFixed(3)).padStart(7)}  ${o.label.padEnd(34)} ko ${o.ko === null ? '-' : o.ko.toFixed(2)}  dead ${o.dead === null ? '-' : o.dead.toFixed(2)}`);
	}
	if (row.dbg) console.log(JSON.stringify(row.dbg, null, 1));
	void captured; void origDecide;
	process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
