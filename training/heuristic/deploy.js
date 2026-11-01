#!/usr/bin/env node
// Checks the bot and, if it passes, puts it where PSBot.exe picks it up.
//
//   node heuristic/deploy.js --check                 run the checks only
//   node heuristic/deploy.js --note "what changed"   check, stamp a build, deploy, wait for the page to load it
//   node heuristic/deploy.js --rollback              put the previous deployed build back
//   node heuristic/deploy.js --list                  the builds deployed so far
//
// The checks are the ones a broken build must not get past: the file parses, it loads headless,
// every position test passes and every lesson from a lost game still comes out right. A deploy
// copies psbot.js (with BUILD stamped in) next to PSBot.exe, which loads it into the open
// Showdown page without a reload; this waits for that to be confirmed and rolls back if the page
// reports that the new build failed to load.
//
// --target <folder> overrides the default C:\Users\you\showdown-bot (or PSBOT_DEPLOY_DIR).
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {spawnSync} = require('child_process');

const ROOT = path.join(__dirname, '..');
const SRC = path.join(ROOT, 'psbot', 'psbot.js');
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : def; };
const has = name => args.includes('--' + name);
const TARGET = opt('target', process.env.PSBOT_DEPLOY_DIR || 'C:\\Users\\you\\showdown-bot');
const VERSIONS = path.join(TARGET, 'versions');
const LEDGER = path.join(TARGET, 'logs', 'deploys.jsonl');
const sleep = ms => new Promise(r => setTimeout(r, ms));

function run(cmd, cmdArgs) {
	const r = spawnSync(cmd, cmdArgs, {cwd: ROOT, encoding: 'utf8', timeout: 10 * 60 * 1000});
	return {ok: r.status === 0, out: (r.stdout || '') + (r.stderr || '')};
}

function checks() {
	const res = {ok: true, steps: []};
	const step = (name, ok, detail) => {
		res.steps.push({name, ok, detail});
		if (!ok) res.ok = false;
		console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
	};
	const src = fs.readFileSync(SRC, 'utf8');
	try { new vm.Script(src, {filename: 'psbot.js'}); step('parses', true); } catch (e) { step('parses', false, e.message); return res; }
	step('has a build stamp to fill', /var BUILD = 'dev';/.test(src));
	const t = run(process.execPath, [path.join(__dirname, 'test.js')]);
	const tm = /(\d+)\/(\d+) position tests pass/.exec(t.out);
	step('position tests', t.ok && !!tm && tm[1] === tm[2], tm ? tm[0] : t.out.trim().split('\n').slice(-3).join(' | '));
	if (!t.ok) console.log(t.out.split('\n').filter(l => !l.startsWith('PASS')).join('\n'));
	const l = run(process.execPath, [path.join(__dirname, 'lessons.js')]);
	const lm = /(\d+)\/(\d+) lessons pass/.exec(l.out);
	step('lessons from lost games', l.ok && !!lm && lm[1] === lm[2], lm ? lm[0] : l.out.trim().split('\n').slice(-3).join(' | '));
	if (!l.ok) console.log(l.out.split('\n').filter(x => !x.startsWith('PASS')).join('\n'));
	res.tests = tm ? tm[0] : '';
	res.lessons = lm ? lm[0] : '';
	return res;
}

function ledger() {
	if (!fs.existsSync(LEDGER)) return [];
	return fs.readFileSync(LEDGER, 'utf8').split('\n').filter(Boolean).map(l => { try { return JSON.parse(l); } catch (e) { return null; } }).filter(Boolean);
}
function note(entry) {
	fs.mkdirSync(path.dirname(LEDGER), {recursive: true});
	fs.appendFileSync(LEDGER, JSON.stringify(entry) + '\n');
}

function stampNow() {
	const d = new Date();
	const p = n => ('0' + n).slice(-2);
	return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

// Writes the file in one step, so PSBot.exe never sees half of it.
function place(text) {
	const dest = path.join(TARGET, 'psbot.js');
	const tmp = dest + '.new';
	fs.writeFileSync(tmp, text);
	fs.renameSync(tmp, dest);
}

async function confirm(since, build) {
	// PSBot.exe writes reload.json once the open page has answered. No exe running means there
	// is nothing to confirm: the file is simply in place for the next start.
	const file = path.join(TARGET, 'reload.json');
	const statusFile = path.join(TARGET, 'status.json');
	for (let i = 0; i < 40; i++) {
		await sleep(1500);
		try {
			const r = JSON.parse(fs.readFileSync(file, 'utf8'));
			if (Date.parse(r.t) >= since) return {seen: true, ok: !!r.ok, outcome: r.outcome};
		} catch (e) {}
		try {
			const s = JSON.parse(fs.readFileSync(statusFile, 'utf8'));
			if (s.build === build && s.ts >= since) return {seen: true, ok: true, outcome: 'status reports ' + build};
		} catch (e) {}
	}
	return {seen: false, ok: true, outcome: 'no answer from a running PSBot.exe (it will load this build when it next starts)'};
}

(async () => {
	if (has('list')) {
		for (const e of ledger()) console.log(`${e.t}  ${e.build}  ${e.action}  ${e.note || ''}  ${e.outcome || ''}`);
		return;
	}
	if (has('rollback')) {
		const done = ledger().filter(e => e.action === 'deploy' && e.ok);
		const cur = done[done.length - 1], prev = done[done.length - 2];
		if (!prev) { console.error('no earlier build to go back to'); process.exit(1); }
		const file = path.join(VERSIONS, `psbot-${prev.build}.js`);
		if (!fs.existsSync(file)) { console.error('missing ' + file); process.exit(1); }
		const since = Date.now();
		place(fs.readFileSync(file, 'utf8'));
		const c = await confirm(since, prev.build);
		note({t: new Date().toISOString(), action: 'rollback', build: prev.build, from: cur && cur.build, note: opt('note', ''), ok: c.ok, outcome: c.outcome});
		// A rolled-back build counts as the current one again.
		note({t: new Date().toISOString(), action: 'deploy', build: prev.build, version: prev.version, note: 'rollback from ' + (cur && cur.build), ok: c.ok, outcome: c.outcome});
		console.log(`rolled back to ${prev.build}: ${c.outcome}`);
		process.exit(c.ok ? 0 : 1);
	}

	const res = checks();
	if (!res.ok) { console.error('\nNot deployed: the checks failed.'); process.exit(1); }
	if (has('check')) { console.log('\nAll checks pass.'); return; }

	const src = fs.readFileSync(SRC, 'utf8');
	const version = (/var VERSION = '([^']+)'/.exec(src) || [])[1] || '?';
	const build = stampNow();
	const stamped = src.replace("var BUILD = 'dev';", `var BUILD = '${build}';`);
	new vm.Script(stamped, {filename: 'psbot.js'});
	fs.mkdirSync(VERSIONS, {recursive: true});
	fs.writeFileSync(path.join(VERSIONS, `psbot-${build}.js`), stamped);
	const since = Date.now();
	place(stamped);
	console.log(`\nplaced ${version} build ${build} in ${TARGET}, waiting for the page to load it`);
	const c = await confirm(since, build);
	console.log(c.outcome);
	note({t: new Date().toISOString(), action: 'deploy', build, version, note: opt('note', ''), tests: res.tests, lessons: res.lessons, ok: c.ok, outcome: c.outcome});
	if (!c.ok) {
		// The page said the new build would not load. It has already put the old one back; make
		// the file on disk agree with what is running.
		const done = ledger().filter(e => e.action === 'deploy' && e.ok);
		const prev = done[done.length - 1];
		if (prev && fs.existsSync(path.join(VERSIONS, `psbot-${prev.build}.js`))) {
			place(fs.readFileSync(path.join(VERSIONS, `psbot-${prev.build}.js`), 'utf8'));
			console.error(`restored ${prev.build} on disk`);
		}
		process.exit(1);
	}
	console.log(`deployed ${version} build ${build}`);
})().catch(e => { console.error(e); process.exit(1); });
