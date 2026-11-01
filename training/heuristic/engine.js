#!/usr/bin/env node
/**
 * Headless PSBot.
 *
 * Runs psbot/psbot.js, the same file the browser bot uses, inside Node. The battle state comes
 * from the Showdown client's own battle code (fetched by setup.js into heuristic/cache), driven
 * with a stub scene, so the heuristic sees exactly what it sees in the page.
 *
 * As a module:   const engine = require('./engine').createEngine();
 * As a server:   node heuristic/engine.js      (one JSON object per line on stdin and stdout)
 *
 *   {"id": 1, "op": "feed",   "key": "bot|battle-gen9randombattle-1", "room": "battle-gen9randombattle-1", "data": "|switch|...\n|turn|1"}
 *   {"id": 2, "op": "decide", "key": "...", "room": "...", "request": {...}}   -> {"id": 2, "ok": true, "choice": "move 1", "table": [...]}
 *   {"id": 3, "op": "chose",  "key": "...", "choice": "switch 3"}              (the choice that was actually sent)
 *   {"id": 4, "op": "end",    "key": "..."}
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const DEFAULT_CACHE = path.join(__dirname, 'cache');
// Load order matters: the dex first, then its data, then the battle code that reads both.
const CLIENT_FILES = [
	'battle-dex-data.js', 'battle-dex.js', 'pokedex.js', 'moves.js', 'abilities.js', 'items.js',
	'typechart.js', 'aliases.js', 'battle-text-parser.js', 'battle-scene-stub.js', 'battle.js',
];
// Only needed for generations before 9, and 15 MB, so it is optional.
const OLD_GEN_TABLES = 'teambuilder-tables.js';

function createEngine(options = {}) {
	const cacheDir = options.cacheDir || process.env.PSBOT_CACHE || DEFAULT_CACHE;
	const botFile = options.botFile || process.env.PSBOT_JS || path.join(ROOT, 'psbot', 'psbot.js');
	const offline = !!options.offline;
	const logTo = options.log || (() => {});

	for (const f of CLIENT_FILES) {
		if (!fs.existsSync(path.join(cacheDir, f))) {
			throw new Error(`Missing ${f} in ${cacheDir}. Run "node heuristic/setup.js" first.`);
		}
	}

	// Set data: serve it from the cache folder, and fill the cache on a miss.
	function cachedFetch(url) {
		const name = String(url).split('/').pop().split('?')[0];
		const file = path.join(cacheDir, name);
		const fail = {ok: false, json: () => Promise.resolve(null)};
		if (fs.existsSync(file)) {
			try {
				const data = JSON.parse(fs.readFileSync(file, 'utf8'));
				return Promise.resolve({ok: true, json: () => Promise.resolve(data)});
			} catch (e) {
				logTo(`bad cache file ${name}: ${e.message}`);
			}
		}
		if (offline || typeof fetch !== 'function') return Promise.resolve(fail);
		return fetch(url).then(async r => {
			if (!r.ok) return fail;
			const text = await r.text();
			const data = JSON.parse(text);
			try { fs.writeFileSync(file, text); } catch (e) {}
			return {ok: true, json: () => Promise.resolve(data)};
		}).catch(() => fail);
	}

	// The bot logs through console.log. Stdout is the protocol channel, so send that elsewhere.
	const quiet = {
		log: (...a) => logTo(a.join(' ')), info: (...a) => logTo(a.join(' ')),
		warn: (...a) => logTo(a.join(' ')), error: (...a) => logTo(a.join(' ')), debug: () => {},
	};
	const sandbox = {console: quiet, setTimeout, clearTimeout, setInterval, clearInterval, fetch: cachedFetch};
	sandbox.window = sandbox;
	sandbox.self = sandbox;
	sandbox.exports = sandbox; // the data files assign onto "exports"
	sandbox.document = {body: null, hidden: false};
	sandbox.navigator = {userAgent: 'node'};
	const ctx = vm.createContext(sandbox);
	const load = file => vm.runInContext(fs.readFileSync(file, 'utf8'), ctx, {filename: file});

	for (const f of CLIENT_FILES) load(path.join(cacheDir, f));
	const oldGens = fs.existsSync(path.join(cacheDir, OLD_GEN_TABLES));
	if (oldGens) load(path.join(cacheDir, OLD_GEN_TABLES));
	load(botFile);

	const PSBot = sandbox.PSBot;
	if (!PSBot || !PSBot._internals || !PSBot._internals.noteChoice) {
		throw new Error(`${botFile} did not load as a headless-capable PSBot`);
	}
	PSBot.config.verbose = false;
	PSBot.config.panel = false;
	Object.assign(PSBot.config, options.config || {});
	// PSBOT_CONFIG='{"teraMargin":0.3}' overrides bot settings, for tuning and A/B runs.
	if (process.env.PSBOT_CONFIG) Object.assign(PSBot.config, JSON.parse(process.env.PSBOT_CONFIG));
	const I = PSBot._internals;
	const rooms = new Map();
	const sleep = ms => new Promise(r => setTimeout(r, ms));

	function drain(battle) {
		// The stub scene has no animations to wait for, so this runs every queued line at once.
		if (!battle.atQueueEnd) battle.seekTurn(Infinity);
	}

	function getRoom(key, roomId) {
		let r = rooms.get(key);
		if (r) return r;
		const id = roomId || key;
		const battle = new sandbox.Battle({id});
		r = {room: {id, battle, request: null, side: '', choice: null}, st: {id, startedAt: Date.now()}};
		rooms.set(key, r);
		return r;
	}

	function setRequest(r, req) {
		r.room.request = req;
		if (r.st.lastRqid !== req.rqid) {
			r.st.lastRqid = req.rqid;
			r.st.errors = 0;
			r.st.badChoices = {};
		}
		const side = req.side && req.side.id;
		if (side && r.room.side !== side) {
			r.room.side = side;
			r.room.battle.setViewpoint(side);
			drain(r.room.battle);
		}
	}

	function feed(key, roomId, data) {
		const r = getRoom(key, roomId);
		const battle = r.room.battle;
		for (const line of String(data).split('\n')) {
			if (!line || line[0] === '>') continue;
			if (line.startsWith('|request|')) {
				const json = line.slice(9);
				if (json) {
					try { setRequest(r, JSON.parse(json)); } catch (e) { logTo(`bad request json in ${key}: ${e.message}`); }
				}
				continue;
			}
			try { battle.add(line); } catch (e) { logTo(`battle.add failed in ${key} on "${line.slice(0, 80)}": ${e.message}`); }
		}
		try { drain(battle); } catch (e) { logTo(`drain failed in ${key}: ${e.message}`); }
		try { I.onRoomData(r.room, r.st, String(data)); } catch (e) {}
	}

	async function ensureSets(r) {
		const battle = r.room.battle;
		const fmt = (r.room.id || '').split('-')[1] || '';
		if (!fmt) return;
		I.loadSets(fmt, battle.gen || 9);
		const t0 = Date.now();
		while (I.setsLoading[fmt] && Date.now() - t0 < 8000) await sleep(5);
		// The bot's own copy of the move data (the client strips moves once they are used).
		while (I.movesBusy && I.movesBusy() && Date.now() - t0 < 8000) await sleep(5);
	}

	const round = x => (typeof x === 'number' && isFinite(x) ? Math.round(x * 1000) / 1000 : null);

	async function decide(key, roomId, request, data) {
		if (data) feed(key, roomId, data);
		const r = getRoom(key, roomId);
		if (request) setRequest(r, request);
		const req = r.room.request;
		if (!req) return {choice: 'default', why: 'no request yet', table: []};
		if (req.wait) return {choice: 'default', why: 'waiting', table: []};
		await ensureSets(r);
		drain(r.room.battle);
		const d = (r.st.errors || 0) >= 3 ?
			{choice: 'default', why: 'fallback after repeated rejections', table: []} :
			I.decide(r.room, req, r.st);
		return {
			choice: d.choice,
			why: d.why,
			turn: r.room.battle.turn,
			table: (d.table || []).map(o => ({
				choice: o.choice, label: o.label, v: round(o.V), ko: round(o.pKO), dead: round(o.pDead), sw: !!o.isSwitch,
				dealt: round(o.dealt), dmg: o.r ? [Math.round(o.r.min), Math.round(o.r.max)] : null, acc: o.r ? round(o.r.acc) : null,
			})),
			dbg: d.dbg || null,
		};
	}

	function chose(key, choice) {
		const r = rooms.get(key);
		if (!r || !r.room.request) return;
		r.st.handledReq = r.room.request;
		I.noteChoice(r.st, r.room.request, String(choice), r.room.battle.turn);
	}

	function end(key) {
		const r = rooms.get(key);
		if (!r) return;
		try { r.room.battle.destroy(); } catch (e) {}
		rooms.delete(key);
	}

	async function handle(msg) {
		switch (msg.op) {
		case 'feed': feed(msg.key, msg.room, msg.data || ''); return {};
		case 'decide': return decide(msg.key, msg.room, msg.request, msg.data);
		case 'chose': chose(msg.key, msg.choice); return {};
		case 'end': end(msg.key); return {};
		case 'ping': return {pong: true, version: PSBot.version, battles: rooms.size, oldGens, predictions: PSBot.predictions};
		default: throw new Error(`unknown op "${msg.op}"`);
		}
	}

	return {feed, decide, chose, end, handle, rooms, sandbox, PSBot, oldGens};
}

function serve() {
	const readline = require('readline');
	const engine = createEngine({log: line => process.stderr.write(`[engine] ${line}\n`)});
	const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
	// One message at a time, in arrival order, so a decide always sees the lines fed before it.
	let chain = Promise.resolve();
	rl.on('line', line => {
		if (!line.trim()) return;
		chain = chain.then(async () => {
			let msg = null;
			let out;
			try {
				msg = JSON.parse(line);
				out = Object.assign({id: msg.id, ok: true}, await engine.handle(msg));
			} catch (e) {
				out = {id: msg ? msg.id : null, ok: false, error: String(e && e.message || e)};
				process.stderr.write(`[engine] ${out.error}\n`);
			}
			// A message without an id is fire-and-forget: nobody is waiting for the answer.
			if (msg && msg.id === undefined) return;
			process.stdout.write(JSON.stringify(out) + '\n');
		});
	});
	rl.on('close', () => chain.then(() => process.exit(0)));
}

module.exports = {createEngine, CLIENT_FILES, OLD_GEN_TABLES, DEFAULT_CACHE};

if (require.main === module) serve();
