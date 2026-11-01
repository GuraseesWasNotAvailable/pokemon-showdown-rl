#!/usr/bin/env node
// Lessons from lost games. Each entry in lessons.json names a recorded game, a turn, and what
// the bot should (or should not) pick there. This replays that turn through the current bot and
// checks it, so a mistake that once cost a game stays fixed.
//
//   node heuristic/lessons.js                 run every lesson
//   node heuristic/lessons.js --only 24       run the lessons whose name contains "24"
//   node heuristic/lessons.js --show          also print the score table for each
//   node heuristic/lessons.js add <game.json> copy a recorded game into heuristic/lessons/ with names removed
'use strict';

const fs = require('fs');
const path = require('path');

const DIR = path.join(__dirname, 'lessons');
const LIST = path.join(__dirname, 'lessons.json');

// Keeps what the replay needs and drops chat, joins, timers and both players' names.
function scrub(rec) {
	const me = rec.me && rec.me.name, foe = rec.opponent && rec.opponent.name;
	const swap = ln => {
		let out = ln;
		if (me) out = out.split(me).join('Me');
		if (foe) out = out.split(foe).join('Foe');
		return out;
	};
	const log = (rec.log || []).filter(ln => !/^\|(c|c:|j|J|l|L|n|N|raw|html|uhtml|uhtmlchange|inactive|inactiveoff|t:|chat|bigerror)\|/.test(ln) && ln !== '|').map(swap);
	return {
		v: rec.v, bot: rec.bot, id: rec.id, format: rec.format, gen: rec.gen, result: rec.result, turns: rec.turns,
		me: {name: 'Me', side: rec.me.side}, opponent: {name: 'Foe'},
		myTeam: rec.myTeam,
		decisions: (rec.decisions || []).map(d => ({turn: d.turn, forced: d.forced, choice: d.choice, why: d.why, mine: d.mine, req: d.req || undefined})),
		log,
	};
}

function add(file) {
	const rec = JSON.parse(fs.readFileSync(file, 'utf8'));
	fs.mkdirSync(DIR, {recursive: true});
	const short = rec.id.replace(/^battle-/, '').split('-').slice(0, 2).join('-');
	const out = path.join(DIR, short + '.json');
	fs.writeFileSync(out, JSON.stringify(scrub(rec)));
	return short;
}

async function run(opts) {
	// The lessons are turns of real public ladder games, which are not published with the code:
	// without them there is nothing to check (add your own with `node lessons.js add <battle.json>`).
	if (!fs.existsSync(LIST)) { console.log('\n0/0 lessons pass (no lessons.json: none recorded here)'); return 0; }
	const {createEngine} = require('./engine');
	const {replay} = require('./replay');
	const lessons = JSON.parse(fs.readFileSync(LIST, 'utf8')).filter(l => !opts.only || (l.name + ' ' + l.game).includes(opts.only));
	const engine = createEngine({log: () => {}});
	let pass = 0, fail = 0;
	for (const l of lessons) {
		const file = path.join(DIR, l.game + '.json');
		if (!fs.existsSync(file)) { console.log(`MISSING ${l.name} (${l.game})`); fail++; continue; }
		const r = await replay(file, {turn: l.turn, all: true, losses: false}, engine);
		const rows = (r ? r.rows : []).filter(x => x.turn === l.turn && !!x.forced === !!l.forced);
		const row = rows[l.nth || 0];
		if (!row) { console.log(`MISSING ${l.name}: no decision on turn ${l.turn}`); fail++; continue; }
		const picked = (row.table || []).find(o => o.choice === row.nowChoice);
		const label = picked ? picked.label : row.nowChoice;
		const hit = rx => new RegExp(rx, 'i').test(label);
		const okWant = !l.want || l.want.some(hit);
		const okAvoid = !l.avoid || !l.avoid.some(hit);
		const ok = okWant && okAvoid;
		if (ok) pass++; else fail++;
		const top = (row.table || []).slice(0, 6).map(o => `${o.label} ${o.v === null ? '?' : o.v.toFixed(2)}`).join(' | ');
		if (!ok || opts.show) {
			console.log(`${ok ? 'PASS' : 'FAIL'}  ${l.name}`);
			console.log(`      picked ${label}${l.want ? '   want ' + l.want.join(' / ') : ''}${l.avoid ? '   avoid ' + l.avoid.join(' / ') : ''}`);
			console.log(`      was ${row.was}`);
			console.log(`      [${top}]`);
			if (opts.show && row.dbg) console.log('      ' + JSON.stringify(row.dbg));
		} else {
			console.log(`PASS  ${l.name}  ->  ${label}`);
		}
	}
	console.log(`\n${pass}/${pass + fail} lessons pass`);
	return fail;
}

if (require.main === module) {
	const args = process.argv.slice(2);
	if (args[0] === 'add') {
		for (const f of args.slice(1)) console.log(add(f));
	} else {
		const oi = args.indexOf('--only');
		run({only: oi >= 0 ? args[oi + 1] : '', show: args.includes('--show')}).then(f => process.exit(f ? 1 : 0)).catch(e => { console.error(e); process.exit(1); });
	}
}

module.exports = {scrub, add, run};
