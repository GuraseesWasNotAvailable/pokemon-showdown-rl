// ==UserScript==
// @name         PSBot - Pokemon Showdown random battle bot
// @namespace    psbot
// @version      1.4.3
// @description  Plays singles random battles in the Showdown client: damage calc, speed and priority checks, switch logic.
// @match        https://play.pokemonshowdown.com/*
// @match        https://*.psim.us/*
// @run-at       document-idle
// @grant        none
// ==/UserScript==

/*
 * PSBot runs inside the classic Pokemon Showdown client page. It reads the same request and
 * battle state the client uses to draw the move buttons, scores every legal move and switch,
 * and sends the choice through the client's own sendDecision.
 *
 *   PSBot.start()                      play every battle you are a player in
 *   PSBot.start({dryRun: true})        log what it would pick, send nothing
 *   PSBot.start({autoSearch: true, format: 'gen9randombattle', maxGames: 5})
 *   PSBot.stop()
 *   PSBot.explain()                    score table for the current decision
 *
 * It never sends chat messages.
 */
(function () {
	'use strict';
	var W = window;
	if (W.PSBot && typeof W.PSBot.stop === 'function') {
		// A copy is already running in this page. Take its state first, then stop it.
		try {
			if (typeof W.PSBot._state === 'function') {
				W.__psbotCarry = W.PSBot._state();
				W.__psbotCarry.version = W.PSBot.version + ' ' + (W.PSBot.build || '');
			}
		} catch (e0) { W.__psbotCarry = null; }
		try { W.PSBot.stop(true); } catch (e) {}
	}

	var VERSION = '1.4.3';
	// Stamped by the deploy script, so every recorded game says exactly which revision played it.
	var BUILD = 'dev';
	var cfg = {
		enabled: false,
		dryRun: false, // decide and log, but do not send
		autoSearch: false, // queue a new game when none is running
		format: 'gen9randombattle',
		maxGames: 0, // 0 = no limit (only applies to autoSearch)
		parallel: 1, // games to keep running at once when auto-queueing (1 to 20)
		settleMs: 700, // battle state must be idle this long before deciding
		thinkMs: 500, // extra delay before sending
		fast: false, // skip battle animations instead of waiting for them
		maxAnimWaitMs: 15000, // skip animations anyway if they run longer than this
		closeFinished: true, // close a finished battle tab once the next game starts or auto-queue is on
		autoTimer: true, // turn the battle timer on at the start of every game
		record: true, // write every game down: teams, decisions, turn history, result, rating
		switchPenalty: 0.1, // tempo cost of a switch, on top of the hit the incoming Pokemon takes
		switchOnlyBelow: 0.8, // never leave a matchup that is still worth this much
		switchMargin: 0.3, // and only for a teammate that scores this much better
		predictSwitch: 0.12, // base weight given to the opponent switching out this turn
		switchPressure: 0.1, // extra weight when its matchup is bad and its bench is better
		predictCap: 0.5, // never weigh a predicted switch above this
		doomedMargin: 0.1, // margin a switch needs when staying in means going down without a KO
		sureKo: 0.06, // extra value for a KO that is all but certain and safe to take
		sweepWeight: 0.75, // how much a boost's value against the rest of their team counts
		dropAware: true, // score attacks that lower the target's stats by what the drop changes
		boostWorth: 0.07, // extra value per boost stage for knocking out a boosted Pokemon
		urgency: 0.12, // cost of a passive turn while the opponent is setting up
		switchEat: 0.6, // how much "takes my obvious move well" counts when naming the switch-in
		predictTera: 0.08, // base weight given to the opponent Terastallizing this turn (0 = ignore)
		predictTeraPickle: 0.15, // the same, when a Tera would blunt the hit I am threatening
		teraMargin: 0.5, // how much better Tera must be to spend it without a KO at stake
		teraEarly: 0.2, // added to that margin on the first turns, when the gain is mostly a guess
		teraKoGain: 0.15, // gain needed to spend it on securing a KO
		teraSurprise: false, // score my Tera turn against the move picked for my old typing
		teraStatus: false, // allow Tera on a turn I use a status move
		deepRace: true, // project healing, setup and stat changes when playing a matchup forward
		setupTrust: 0.85, // how sure I am the opponent uses a setup move it has when that is its best line
		healTrust: 0.9, // the same for a recovery move
		takeKo: true, // a KO that is all but certain beats healing, setting up or switching
		switchStayTax: 0.6, // extra margin a switch needs per point of value staying in still has
		sleepMargin: 0.02, // margin a switch needs when my active Pokemon is asleep or frozen
		lockPenalty: 0.15, // cost of a move that locks me in (Outrage) while they still have a bench
		revenge: 0.45, // when the active is about to go down anyway: how much the free switch after the faint counts against switching now
		entryRisk: true, // score a switch by the chance the hit on the way in kills, not by an average hit
		wideRange: false, // damage ranges cover every item and ability the attacker might have (measured worse head to head, so off)
		switchRelief: 0.16, // how much less a switch has to gain when staying in is clearly losing
		teraSave: true, // spend Tera when it turns a death this turn into a hit the Pokemon survives
		freeSetup: 0.5, // chance a Pokemon with a setup move uses it on the turn I switch (times how sure the move is in its set)
		keepAnswers: 0.5, // extra cost of HP lost by the one Pokemon that beats a threat waiting on their bench (0 = off)
		teraCal: true, // chance of their Tera this turn from what 390 ladder games showed (the old guess was four times too high)
		swCal: true, // their switch weight corrected where the ladder games showed it low: when I threaten the KO, and when their stats are lowered
		restNap: true, // Rest in a projection: a full heal, then two turns asleep (it used to be a free 60% heal)
		phazeTeam: 0.35, // share of what a boosted sweeper would do to the rest of my team that counts toward Roar, Whirlwind and Haze
		// A ladder experiment. Each live game is put in one of two arms by its battle id: `new` plays
		// with the settings above, `base` with the overrides below, and the arm goes into the
		// game's record. That is the only way to tell from ladder results whether a change to the
		// scoring helps, because a streak looks the same whatever caused it. null turns it off.
		arms: {id: 'r3', share: 0.4, base: {freeSetup: 0, phazeTeam: 0, restNap: false, entryRisk: false, revenge: 0, switchRelief: 0, teraSave: false, teraCal: false, swCal: false, keepAnswers: 0}},
		watchdog: true, // recover on its own from a dropped connection, a stuck search or a stuck turn
		reloadHours: 3, // while auto-queueing, reload the page this often (between games) to keep it fresh
		resumeMinutes: 10, // after a reload or a crash, pick auto-queue back up if it was on this recently
		verbose: true,
		panel: true
	};
	var record = {wins: 0, losses: 0, ties: 0, games: 0, searches: 0};
	var rooms = {}; // roomid -> per-battle bot state
	// What a long unattended run needs to know about itself.
	var run = {loadedAt: Date.now(), reloads: 0, reloadTimes: [], lastBeat: 0, lastLogFlush: 0, downSince: 0, reloading: false,
		decisions: 0, lastDecisionAt: 0, lastTickAt: 0, errors: 0, events: 0, resumed: false, lastPrune: 0, slowest: 0};
	var logOut = []; // log lines not yet written to disk
	var timer = null;
	var logBuf = [];
	var lastExplain = null;
	var lastSearchAt = 0;
	var lastSearchLive = 0; // games running when the last search was sent
	var queueBase = 0; // record.searches when auto-queue was last switched on
	var MAX_PARALLEL = 20; // most games the bot will run at once
	// Ladder rating after each rated game, kept in localStorage so the graph survives a reload.
	var elo = {history: [], loadedFor: null, sessionStart: null};
	// How the switch predictions turned out: made, opponent did switch, and it was the Pokemon named.
	var predictions = {made: 0, switched: 0, rightMon: 0, weight: 0};

	// ------------------------------------------------------------------ utils

	function toID(s) {
		return ('' + (s === null || s === undefined ? '' : s)).toLowerCase().replace(/[^a-z0-9]+/g, '');
	}
	function clampInt(v, lo, hi) {
		v = Math.round(+v);
		if (!isFinite(v)) v = lo;
		return Math.max(lo, Math.min(hi, v));
	}
	function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
	function uniq(a) { return Array.from(new Set(a)); }
	function clone(o, extra) { return Object.assign({}, o, extra || {}); }
	function pct(x) { return Math.round(x * 100) + '%'; }
	function log() {
		var msg = Array.prototype.slice.call(arguments).map(function (x) {
			return typeof x === 'string' ? x : JSON.stringify(x);
		}).join(' ');
		logBuf.push('[' + new Date().toLocaleTimeString() + '] ' + msg);
		if (logBuf.length > 400) logBuf.shift();
		logOut.push(new Date().toISOString() + ' ' + msg);
		if (logOut.length > 3000) logOut.splice(0, 1000);
		if (ui) ui.logVer++;
		if (cfg.verbose) console.log('[PSBot] ' + msg);
		renderPanel();
	}

	// ------------------------------------------------------------------ disk channel and run state

	// PSBot.exe exposes psbotSave(json) to the page. Everything that should outlive the tab goes
	// through it: finished games, a status heartbeat, the log, and one line per notable event.
	function toDisk(kind, payload) {
		try {
			if (typeof W.psbotSave !== 'function') return false;
			W.psbotSave(JSON.stringify(Object.assign({kind: kind}, payload)));
			return true;
		} catch (e) { return false; }
	}
	function localDay() {
		var d = new Date();
		return d.getFullYear() + '-' + ('0' + (d.getMonth() + 1)).slice(-2) + '-' + ('0' + d.getDate()).slice(-2);
	}
	function event(type, detail) {
		run.events++;
		var text = typeof detail === 'string' ? detail : JSON.stringify(detail || {});
		toDisk('event', {line: JSON.stringify({t: new Date().toISOString(), type: type, detail: detail || null, v: VERSION, build: BUILD})});
		log('[' + type + '] ' + text);
	}
	function flushLog() {
		if (!logOut.length) return;
		if (toDisk('log', {day: localDay(), text: logOut.join('\n')})) logOut.length = 0;
	}
	function statusNow() {
		var named = false, uid = '', down = false, mem = null, last = null;
		try { named = !!W.app.user.get('named'); uid = W.app.user.get('userid'); down = !!W.app.isDisconnected; } catch (e) {}
		try { mem = Math.round(W.performance.memory.usedJSHeapSize / 1048576); } catch (e2) {}
		try { var sr = eloSeries(); if (sr) last = sr.points[sr.points.length - 1]; } catch (e3) {}
		return {
			t: new Date().toISOString(), ts: Date.now(), version: VERSION, build: BUILD, enabled: cfg.enabled, auto: cfg.autoSearch, dryRun: cfg.dryRun,
			format: cfg.format, parallel: cfg.parallel, live: ui.live, searching: ui.searching, named: named, user: uid, disconnected: down,
			record: record, elo: last, eloStart: elo.sessionStart, uptimeMin: Math.round((Date.now() - run.loadedAt) / 60000), reloads: run.reloads,
			decisions: run.decisions, lastDecisionAt: run.lastDecisionAt, slowestMs: run.slowest, errors: run.errors, events: run.events, heapMB: mem,
			draining: !!ui.draining, blockedUntil: ui.blockedUntil, serverCap: ui.serverCap, rooms: Object.keys(rooms).length,
			predictions: predictions, movesFull: pristine.full, recorder: recorder
		};
	}

	// Enough about the run to pick it back up after a page reload, a hot update or a crash.
	var RUN_KEY = 'psbot-run';
	function saveRun() {
		try {
			W.localStorage.setItem(RUN_KEY, JSON.stringify({
				ts: Date.now(), auto: !!cfg.autoSearch && !!cfg.enabled && !cfg.dryRun, format: cfg.format, parallel: cfg.parallel, maxGames: cfg.maxGames,
				record: record, queueBase: queueBase, eloStart: elo.sessionStart, reloads: run.reloads, reloadTimes: run.reloadTimes.slice(-20), predictions: predictions
			}));
		} catch (e) {}
	}
	function loadRun() {
		try {
			var d = JSON.parse(W.localStorage.getItem(RUN_KEY) || 'null');
			return d && typeof d.ts === 'number' ? d : null;
		} catch (e) { return null; }
	}
	function reloadPage(why) {
		if (run.reloading) return;
		run.reloading = true;
		run.reloads++;
		run.reloadTimes.push(Date.now());
		saveRun();
		event('reload', why);
		flushLog();
		setTimeout(function () { try { W.location.reload(); } catch (e) {} }, 400);
	}

	// A game in progress keeps its decisions in localStorage, so the record survives a reload.
	// The experiment arm of a live game, fixed by its id so a reload or a hot update keeps it.
	function armOf(id) {
		var a = cfg.arms;
		if (!a || !a.base || !/^battle-/.test(id || '')) return '';
		var str = id + '|' + a.id, h = 2166136261;
		for (var i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
		return (h % 1000) / 1000 < (a.share === undefined ? 0.5 : a.share) ? 'base' : 'new';
	}

	function recKey(id) { return 'psbot-rec-' + id; }
	function stashRec(st) {
		if (!st || !st.rec || !st.id) return;
		try { W.localStorage.setItem(recKey(st.id), JSON.stringify({ts: Date.now(), startedAt: st.startedAt, rec: st.rec})); } catch (e) {}
	}
	function unstashRec(id) {
		try {
			var d = JSON.parse(W.localStorage.getItem(recKey(id)) || 'null');
			return d && d.rec && Array.isArray(d.rec.decisions) ? d : null;
		} catch (e) { return null; }
	}
	function dropRec(id) { try { W.localStorage.removeItem(recKey(id)); } catch (e) {} }
	function sweepRecs() {
		try {
			var ls = W.localStorage, old = [];
			for (var i = 0; i < ls.length; i++) {
				var k = ls.key(i);
				if (!k || k.indexOf('psbot-rec-') !== 0) continue;
				var d = null;
				try { d = JSON.parse(ls.getItem(k)); } catch (e) {}
				if (!d || !d.ts || Date.now() - d.ts > 12 * 3600 * 1000) old.push(k);
			}
			old.forEach(function (k2) { ls.removeItem(k2); });
		} catch (e2) {}
	}

	// ------------------------------------------------------------------ dex access

	function dexOf(battle) { return (battle && battle.dex) || W.Dex; }

	// The client replaces each BattleMovedex entry with a slimmed-down Move object the first time
	// it looks that move up (which it does for every move used in a battle), and that object
	// drops what the bot needs: boosts, self, drain, selfSwitch, sideCondition and more. So the
	// bot keeps its own untouched copy: whatever is still intact when it loads, topped up from
	// the full data file. It lives on the window so a reloaded bot inherits it.
	var pristine = W.__psbotMoves || (W.__psbotMoves = {full: false, data: {}});
	var movesLoading = 0;
	var moveCache = {};
	function intact(d) { return !!d && typeof d === 'object' && typeof d.exists !== 'boolean'; }
	function snapshotMoves() {
		var table = W.BattleMovedex || {};
		for (var id in table) if (!pristine.data[id] && intact(table[id])) pristine.data[id] = table[id];
	}
	function loadMoves() {
		snapshotMoves();
		if (pristine.full || movesLoading || typeof fetch !== 'function') return;
		movesLoading = Date.now();
		fetch('https://play.pokemonshowdown.com/data/moves.json').then(function (r) {
			return r.ok ? r.json() : null;
		}).then(function (d) {
			movesLoading = 0;
			if (!d || !d.tackle) return;
			for (var id in d) pristine.data[id] = d[id];
			pristine.full = true;
			moveCache = {};
			log('Loaded full move data (' + Object.keys(d).length + ' moves)');
		}).catch(function () { movesLoading = 0; });
	}
	function getMove(battle, rawId) {
		var id = toID(rawId);
		var gen = (battle && battle.gen) || 9;
		var key = gen + ':' + id;
		if (moveCache[key]) return moveCache[key];
		var table = W.BattleMovedex || {};
		var raw = pristine.data[id] || (intact(table[id]) ? (pristine.data[id] = table[id]) : table[id]);
		if (!raw) {
			// "hiddenpowerice70", "return102"
			var stripped = id.replace(/\d+$/, '');
			if (pristine.data[stripped] || table[stripped]) { id = stripped; raw = pristine.data[stripped] || table[stripped]; }
		}
		var mv = Object.assign({basePower: 0, category: 'Status', type: 'Normal', accuracy: true, priority: 0}, raw || {});
		mv.id = id;
		try {
			var dm = dexOf(battle).moves.get(id);
			if (dm && dm.exists !== false && dm.name) {
				mv.name = dm.name;
				mv.type = dm.type;
				mv.category = dm.category;
				if (typeof dm.basePower === 'number') mv.basePower = dm.basePower;
				if (dm.accuracy !== undefined) mv.accuracy = dm.accuracy;
				if (typeof dm.priority === 'number') mv.priority = dm.priority;
			}
		} catch (e) {}
		mv.name = mv.name || id;
		mv.flags = mv.flags || {};
		// A stripped entry is all there is until the full data arrives: use it, do not keep it.
		if (intact(raw) || pristine.full || !raw) moveCache[key] = mv;
		return mv;
	}

	function getSpecies(battle, name) {
		var sp = null;
		try { sp = dexOf(battle).species.get(name); } catch (e) {}
		if (!sp || !sp.baseStats) { try { sp = W.Dex.species.get(name); } catch (e2) {} }
		return sp || {name: name, id: toID(name), types: ['Normal'], baseStats: {hp: 80, atk: 80, def: 80, spa: 80, spd: 80, spe: 80}, weightkg: 50, abilities: {}};
	}

	function typeMult(battle, atkType, defType) {
		var dt = null;
		try {
			var t = dexOf(battle).types.get(defType);
			dt = t && t.damageTaken;
		} catch (e) {}
		if (!dt) {
			var c = W.BattleTypeChart && W.BattleTypeChart[toID(defType)];
			dt = c && c.damageTaken;
		}
		if (!dt) return 1;
		var v = dt[atkType];
		return v === 1 ? 2 : v === 2 ? 0.5 : v === 3 ? 0 : 1;
	}

	// ------------------------------------------------------------------ random-battle set data

	var setsCache = {}; // format id -> {speciesid: entry} | null
	var setsLoading = {};
	function loadSets(fmt, gen) {
		if (!fmt || fmt in setsCache) return;
		setsCache[fmt] = null;
		if (fmt.indexOf('random') < 0) return;
		setsLoading[fmt] = Date.now();
		var tries = uniq([fmt, 'gen' + gen + 'randombattle']);
		var attempt = function (i) {
			if (i >= tries.length) { delete setsLoading[fmt]; return; }
			fetch('https://pkmn.github.io/randbats/data/' + tries[i] + '.json').then(function (r) {
				return r.ok ? r.json() : null;
			}).then(function (d) {
				if (!d) return attempt(i + 1);
				var idx = {};
				for (var k in d) idx[toID(k)] = d[k];
				setsCache[fmt] = idx;
				delete setsLoading[fmt];
				log('Loaded set data for ' + tries[i] + ' (' + Object.keys(idx).length + ' species)');
			}).catch(function () { attempt(i + 1); });
		};
		attempt(0);
	}
	function setFor(fmt, species) {
		var data = setsCache[fmt];
		if (!data || !species) return null;
		return data[species.id] || data[toID(species.name)] || data[toID(species.baseSpecies)] || null;
	}

	// ------------------------------------------------------------------ parsing

	function parseCond(c) {
		var m = /^(\d+)(?:\/(\d+))?(?:\s+(\w+))?/.exec(c || '');
		if (!m) return {hp: 0, maxhp: 0, status: '', fainted: true};
		var st = m[3] || '';
		return {hp: +m[1], maxhp: m[2] ? +m[2] : 0, status: st === 'fnt' ? '' : st, fainted: st === 'fnt' || +m[1] === 0};
	}
	function parseDetails(d) {
		var parts = ('' + d).split(', ');
		var level = 100;
		for (var i = 1; i < parts.length; i++) {
			if (/^L\d+$/.test(parts[i])) level = +parts[i].slice(1);
		}
		return {species: parts[0], level: level};
	}

	function boostMult(stage) {
		stage = clamp(stage || 0, -6, 6);
		return stage >= 0 ? (2 + stage) / 2 : 2 / (2 - stage);
	}
	function accMult(stage) {
		stage = clamp(stage || 0, -6, 6);
		return stage >= 0 ? (3 + stage) / 3 : 3 / (3 - stage);
	}

	// Opposing stats are not sent by the server. Every Pokemon in a random format shares one
	// EV/IV spread, so we solve for it from our own team's real stats and reuse it.
	function calibrate(battle, req) {
		var xs = [];
		(req.side.pokemon || []).forEach(function (p) {
			var det = parseDetails(p.details);
			var sp = getSpecies(battle, det.species);
			if (!sp.baseStats || !p.stats) return;
			['atk', 'def', 'spa', 'spd', 'spe'].forEach(function (s) {
				var x = (p.stats[s] - 5) * 100 / det.level - 2 * sp.baseStats[s];
				if (isFinite(x)) xs.push(x);
			});
		});
		if (!xs.length) return battle.gen >= 3 ? 52 : 93;
		xs.sort(function (a, b) { return a - b; });
		return clamp(xs[Math.floor(xs.length / 2)], 0, 110);
	}

	function isGrounded(field, c) {
		if (field.gravity) return true;
		var v = c.volatiles || {};
		if (v.smackdown || v.ingrain) return true;
		if (c.item === 'ironball') return true;
		if (c.abilityKnown && c.ability === 'levitate') return false;
		if (v.magnetrise || v.telekinesis) return false;
		if (c.item === 'airballoon') return false;
		return c.types.indexOf('Flying') < 0;
	}

	function mkMine(ctx, p, cm, slot) {
		var det = parseDetails(p.details);
		var cond = parseCond(p.condition);
		var species = getSpecies(ctx.battle, det.species);
		var tera = p.terastallized || '';
		// Types come from the client's own model of my active Pokemon (it tracks Tera, Soak,
		// Protean and so on). The raw request carries no species data, so never read them from it.
		var types = null;
		if (cm) { try { types = cm.getTypeList().slice(); } catch (e) {} }
		if (!types || !types.length || types.indexOf('???') >= 0) types = tera && tera !== 'Stellar' ? [tera] : (species.types || ['Normal']).slice();
		var c = {
			mine: true, name: det.species, ident: p.ident || '', species: species, level: det.level, slot: slot,
			types: types, origTypes: (species.types || []).slice(), tera: tera, teraType: p.teraType || '',
			stats: Object.assign({atk: 100, def: 100, spa: 100, spd: 100, spe: 100}, p.stats || {}),
			hp: cond.hp, maxhp: cond.maxhp || 1, status: cond.status, fainted: cond.fainted,
			boosts: cm ? Object.assign({}, cm.boosts) : {},
			volatiles: cm ? (cm.volatiles || {}) : {},
			ability: ctx.gen < 3 ? '' : toID(p.ability || p.baseAbility), abilityKnown: true, possibleAbilities: [],
			item: ctx.gen < 2 ? '' : toID(p.item), itemKnown: true, possibleItems: [],
			moves: (p.moves || []).map(toID), pool: [], hasSet: true,
			lastMove: cm ? toID(cm.lastMove) : '',
			weightkg: species.weightkg || 50,
			timesAttacked: cm ? (cm.timesAttacked || 0) : 0,
			toxicTurns: cm && cm.statusData ? (cm.statusData.toxicTurns || 0) : 0,
			dynamaxed: !!(cm && cm.volatiles && cm.volatiles.dynamax)
		};
		c.benched = !cm;
		try { c.forme = cm ? toID(cm.getSpeciesForme ? cm.getSpeciesForme() : cm.speciesForme) : ''; } catch (e2) { c.forme = ''; }
		c.grounded = isGrounded(ctx.field, c);
		return c;
	}

	function mkOpp(ctx, cm) {
		var species = cm.getSpecies();
		var level = cm.level || 100;
		var term = ctx.term;
		var bs = species.baseStats || {hp: 80, atk: 80, def: 80, spa: 80, spd: 80, spe: 80};
		var stat = function (b) { return Math.floor((2 * b + term) * level / 100) + 5; };
		var maxhp = bs.hp === 1 ? 1 : Math.floor((2 * bs.hp + term) * level / 100) + level + 10;
		var vol = cm.volatiles || {};
		if (vol.dynamax) maxhp *= 2;
		var frac = cm.maxhp ? cm.hp / cm.maxhp : 0;
		var set = setFor(ctx.fmt, species);
		var revealed = uniq((cm.moveTrack || []).map(function (m) {
			return toID(('' + m[0]).replace(/^\*/, ''));
		}).filter(Boolean));
		var pool = [], possItems = [], possAbil = [], possTera = [];
		// How likely each unrevealed move and each item is, from the sets this Pokemon can run.
		var poolW = {}, itemW = {};
		if (set) {
			var src = [set];
			if (set.roles) {
				var all = Object.keys(set.roles).map(function (k) { return set.roles[k]; });
				var ok = all.filter(function (r) {
					return revealed.every(function (m) {
						return (r.moves || []).some(function (x) { return toID(x) === m; });
					});
				});
				src = ok.length ? ok : all;
			}
			var slots = Math.max(0, 4 - revealed.length);
			src.forEach(function (r) {
				var rMoves = (r.moves || []).map(toID);
				var cand = rMoves.filter(function (m) { return revealed.indexOf(m) < 0; });
				var pIn = cand.length ? Math.min(1, slots / cand.length) : 0;
				rMoves.forEach(function (m) { pool.push(m); });
				cand.forEach(function (m) { poolW[m] = (poolW[m] || 0) + pIn / src.length; });
				var rItems = (r.items || set.items || []).map(toID);
				rItems.forEach(function (i) { possItems.push(i); itemW[i] = (itemW[i] || 0) + 1 / (src.length * rItems.length); });
				(r.abilities || set.abilities || []).forEach(function (a) { possAbil.push(toID(a)); });
				(r.teraTypes || set.teraTypes || []).forEach(function (t) { possTera.push(t); });
			});
		}
		if (!possAbil.length && species.abilities) {
			for (var k in species.abilities) possAbil.push(toID(species.abilities[k]));
		}
		possAbil = uniq(possAbil);
		possItems = uniq(possItems);
		pool = revealed.length >= 4 ? [] : uniq(pool).filter(function (m) { return revealed.indexOf(m) < 0; });

		var ability = toID(cm.ability), abilityKnown = !!ability;
		if (!abilityKnown && possAbil.length === 1) { ability = possAbil[0]; abilityKnown = true; }
		if (ctx.gen < 3 || vol.gastroacid) { ability = ''; abilityKnown = true; possAbil = []; }
		var item = toID(cm.item), itemKnown = !!item || !!cm.prevItem;
		if (!itemKnown && possItems.length === 1) { item = possItems[0]; itemKnown = true; }
		if (ctx.gen < 2) { item = ''; itemKnown = true; possItems = []; }

		var types = ['Normal'], origTypes = species.types || ['Normal'];
		try { types = cm.getTypeList().slice(); } catch (e) { types = origTypes.slice(); }
		try { origTypes = cm.getTypes(undefined, true)[0].slice(); } catch (e2) {}

		// Two different moves since it came in rules out a Choice item.
		var since = ctx.st && ctx.st.oppSince && ctx.st.oppSince.ident === cm.ident ? ctx.st.oppSince.moves : null;
		if (!itemKnown && since && since.length >= 2) {
			var keep = 0;
			for (var ck in itemW) { if (/^choice(band|specs|scarf)$/.test(ck)) delete itemW[ck]; else keep += itemW[ck]; }
			if (keep > 0) for (var ck2 in itemW) itemW[ck2] /= keep;
			possItems = possItems.filter(function (i) { return !/^choice(band|specs|scarf)$/.test(i); });
		}
		var c = {
			mine: false, name: species.name, ident: cm.ident || '', sinceIn: since ? since.slice() : null, species: species, level: level, slot: 0,
			types: types, origTypes: origTypes, tera: cm.terastallized || '', teraType: '',
			stats: {atk: stat(bs.atk), def: stat(bs.def), spa: stat(bs.spa), spd: stat(bs.spd), spe: stat(bs.spe)},
			hp: frac > 0 ? Math.max(1, Math.round(maxhp * frac)) : 0, maxhp: maxhp,
			status: cm.status || '', fainted: !!cm.fainted || !(cm.hp > 0),
			boosts: Object.assign({}, cm.boosts), volatiles: vol,
			ability: ability, abilityKnown: abilityKnown, possibleAbilities: possAbil,
			item: item, itemKnown: itemKnown, possibleItems: possItems,
			moves: revealed, pool: pool, poolW: poolW, itemW: itemKnown ? {} : itemW, hasSet: !!set, possTera: uniq(possTera),
			lastMove: toID(cm.lastMove),
			weightkg: species.weightkg || 50,
			timesAttacked: cm.timesAttacked || 0,
			toxicTurns: cm.statusData ? (cm.statusData.toxicTurns || 0) : 0,
			dynamaxed: !!vol.dynamax
		};
		// Random-battle sets with Trick Room or Gyro Ball are built with no Speed at all.
		if (set && (revealed.indexOf('trickroom') >= 0 || revealed.indexOf('gyroball') >= 0 || (poolW.trickroom || 0) >= 0.99 || (poolW.gyroball || 0) >= 0.99)) {
			c.stats.spe = Math.floor(2 * bs.spe * level / 100) + 5;
		}
		try {
			var farA = ctx.battle.farSide && ctx.battle.farSide.active ? ctx.battle.farSide.active[0] : null;
			c.benched = !!farA && farA !== cm;
			c.forme = toID(cm.getSpeciesForme ? cm.getSpeciesForme() : cm.speciesForme);
		} catch (e3) { c.benched = false; c.forme = ''; }
		c.choiceP = 0;
		if (itemKnown) c.choiceP = /^choice(band|specs|scarf)$/.test(item) ? 1 : 0;
		else for (var ik in itemW) if (/^choice(band|specs|scarf)$/.test(ik)) c.choiceP += itemW[ik];
		c.grounded = isGrounded(ctx.field, c);
		return c;
	}

	function mkField(battle) {
		var pw = function (name) {
			try { return !!battle.hasPseudoWeather(name); } catch (e) { return false; }
		};
		var weather = toID(battle.weather);
		try {
			if (battle.abilityActive && battle.abilityActive(['Air Lock', 'Cloud Nine'])) weather = '';
		} catch (e) {}
		return {
			gen: battle.gen || 9,
			weather: weather,
			sun: weather === 'sunnyday' || weather === 'desolateland',
			rain: weather === 'raindance' || weather === 'primordialsea',
			sand: weather === 'sandstorm',
			snow: weather === 'snow' || weather === 'snowscape',
			hail: weather === 'hail',
			terrain: pw('Electric Terrain') ? 'electric' : pw('Grassy Terrain') ? 'grassy' : pw('Psychic Terrain') ? 'psychic' : pw('Misty Terrain') ? 'misty' : '',
			trickRoom: pw('Trick Room'),
			gravity: pw('Gravity'),
			mySide: (battle.nearSide && battle.nearSide.sideConditions) || {},
			oppSide: (battle.farSide && battle.farSide.sideConditions) || {}
		};
	}
	function sideOf(ctx, c) { return c.mine ? ctx.field.mySide : ctx.field.oppSide; }

	function hasAb(c) {
		if (!c.abilityKnown) return false;
		for (var i = 1; i < arguments.length; i++) if (c.ability === arguments[i]) return true;
		return false;
	}
	// Probability the Pokemon has one of these abilities, given what has been revealed.
	function abChance(c, ids) {
		if (c.abilityKnown) return ids.indexOf(c.ability) >= 0 ? 1 : 0;
		var poss = c.possibleAbilities || [];
		if (!poss.length) return 0;
		return poss.filter(function (a) { return ids.indexOf(a) >= 0; }).length / poss.length;
	}

	function effSpeed(ctx, c) {
		var f = ctx.field;
		var s = (c.stats.spe || 1) * boostMult(c.boosts.spe);
		if (c.status === 'par' && !hasAb(c, 'quickfeet')) s *= f.gen >= 7 ? 0.5 : 0.25;
		if (c.item === 'choicescarf') s *= 1.5;
		if (c.item === 'ironball' || c.item === 'machobrace') s *= 0.5;
		// Regigigas is at half Speed (and half Attack) for its first five turns out.
		if (c.abilityKnown && c.ability === 'slowstart' && ((c.volatiles || {}).slowstart || c.benched)) s *= 0.5;
		if (c.abilityKnown) {
			var a = c.ability;
			if ((a === 'swiftswim' && f.rain) || (a === 'chlorophyll' && f.sun) || (a === 'sandrush' && f.sand) ||
				(a === 'slushrush' && (f.snow || f.hail)) || (a === 'surgesurfer' && f.terrain === 'electric')) s *= 2;
			if (a === 'quickfeet' && c.status) s *= 1.5;
		}
		var v = c.volatiles || {};
		if (v.protosynthesisspe || v.quarkdrivespe) s *= 1.5;
		var sc = sideOf(ctx, c);
		if (sc && sc.tailwind) s *= 2;
		return s;
	}

	// ------------------------------------------------------------------ damage calculation

	var TYPE_ITEMS = {
		charcoal: 'Fire', mysticwater: 'Water', miracleseed: 'Grass', magnet: 'Electric', nevermeltice: 'Ice',
		blackbelt: 'Fighting', poisonbarb: 'Poison', softsand: 'Ground', sharpbeak: 'Flying', twistedspoon: 'Psychic',
		silverpowder: 'Bug', hardstone: 'Rock', spelltag: 'Ghost', dragonfang: 'Dragon', blackglasses: 'Dark',
		metalcoat: 'Steel', silkscarf: 'Normal', fairyfeather: 'Fairy',
		flameplate: 'Fire', splashplate: 'Water', meadowplate: 'Grass', zapplate: 'Electric', icicleplate: 'Ice',
		fistplate: 'Fighting', toxicplate: 'Poison', earthplate: 'Ground', skyplate: 'Flying', mindplate: 'Psychic',
		insectplate: 'Bug', stoneplate: 'Rock', spookyplate: 'Ghost', dracoplate: 'Dragon', dreadplate: 'Dark',
		ironplate: 'Steel', pixieplate: 'Fairy'
	};
	var IMMUNE_ABILITIES = {
		Fire: ['flashfire', 'wellbakedbody'], Water: ['waterabsorb', 'stormdrain', 'dryskin'],
		Electric: ['voltabsorb', 'lightningrod', 'motordrive'], Grass: ['sapsipper'], Ground: ['eartheater']
	};
	var ONE_TIME_MOVES = {fakeout: 1, firstimpression: 1};
	var NO_DAMAGE_MOVES = {counter: 1, mirrorcoat: 1, metalburst: 1, comeuppance: 1, bide: 1, focuspunch: 1, belch: 1, lastresort: 1, naturalgift: 1, fling: 1, spitup: 1, synchronoise: 1, steelroller: 1, dreameater: 1, snore: 1};
	// Moves that need the user to be a certain type, and take that type away when they hit. A
	// Terastallized user keeps it, so the move can be used every turn.
	var TYPE_BURN_MOVES = {doubleshock: 'Electric', burnup: 'Fire'};

	// Abilities that change how hard a move hits. When the attacker's ability is not known yet and
	// one of these is possible, the damage is averaged over the abilities it can have.
	var DAMAGE_ABILITIES = {};
	('hugepower purepower guts hustle gorillatactics orichalcumpulse solarpower hadronengine adaptability technician toughclaws ' +
		'sheerforce ironfist strongjaw megalauncher sharpness reckless punkrock tintedlens neuroforce waterbubble steelworker ' +
		'steelyspirit transistor dragonsmaw rockypayload sandforce analytic aerilate pixilate refrigerate galvanize normalize ' +
		'supremeoverlord toxicboost flareboost skilllink moldbreaker teravolt turboblaze scrappy mindseye noguard compoundeyes ' +
		'protean libero blaze torrent overgrow swarm sniper parentalbond liquidvoice').split(' ').forEach(function (a) { DAMAGE_ABILITIES[a] = 1; });

	// What an item does to the damage of one move.
	function itemPower(item, cat, type, eff, att) {
		if (item === 'choiceband') return cat === 'Physical' ? 1.5 : 1;
		if (item === 'choicespecs') return cat === 'Special' ? 1.5 : 1;
		if (item === 'lifeorb') return 1.3;
		if (item === 'expertbelt') return eff > 1 ? 1.2 : 1;
		if (item === 'lightball') return (att.species.id || '').indexOf('pikachu') === 0 ? 2 : 1;
		if (TYPE_ITEMS[item] === type) return 1.2;
		return 1;
	}

	function weightPower(kg) {
		return kg < 10 ? 20 : kg < 25 ? 40 : kg < 50 ? 60 : kg < 100 ? 80 : kg < 200 ? 100 : 120;
	}

	function maxPower(mv) {
		if (mv.maxMove && mv.maxMove.basePower) return mv.maxMove.basePower;
		var bp = mv.basePower || 0;
		var weak = mv.type === 'Fighting' || mv.type === 'Poison';
		if (weak) return bp >= 150 ? 100 : bp >= 110 ? 95 : bp >= 75 ? 90 : bp >= 65 ? 85 : bp >= 55 ? 80 : bp >= 45 ? 75 : 70;
		return bp >= 150 ? 150 : bp >= 110 ? 140 : bp >= 75 ? 130 : bp >= 65 ? 120 : bp >= 55 ? 110 : bp >= 45 ? 100 : 90;
	}
	function asMax(mv) {
		if (mv.isMaxed) return mv;
		if (mv.category === 'Status') {
			return {id: 'maxguard', name: 'Max Guard', category: 'Status', type: 'Normal', basePower: 0, accuracy: true, priority: 4, flags: {}, stallingMove: true, isMaxed: true};
		}
		return {
			id: mv.id, name: 'Max ' + mv.name, category: mv.category, type: mv.type, basePower: maxPower(mv),
			accuracy: true, priority: 0, flags: {}, isMaxed: true
		};
	}
	function asZ(mv) {
		var bp = mv.zMove && mv.zMove.basePower;
		if (!bp) {
			var b = mv.basePower || 0;
			bp = b >= 140 ? 200 : b >= 130 ? 195 : b >= 120 ? 190 : b >= 110 ? 185 : b >= 100 ? 180 : b >= 90 ? 175 : b >= 80 ? 160 : b >= 70 ? 140 : b >= 60 ? 120 : 100;
		}
		return {id: mv.id, name: 'Z-' + mv.name, category: mv.category, type: mv.type, basePower: bp, accuracy: true, priority: 0, flags: {}, isZ: true};
	}

	function stabMult(att, type, aAb) {
		var orig = att.origTypes && att.origTypes.length ? att.origTypes : att.types;
		var s = 1;
		if (att.tera && att.tera !== 'Stellar') {
			if (att.tera === type) s = orig.indexOf(type) >= 0 ? 2 : 1.5;
			else if (orig.indexOf(type) >= 0) s = 1.5;
			if (aAb === 'adaptability' && att.tera === type) s = orig.indexOf(type) >= 0 ? 2.25 : 2;
		} else if (att.types.indexOf(type) >= 0) {
			s = aAb === 'adaptability' ? 2 : 1.5;
			if (att.tera === 'Stellar') s = 2;
		} else if (att.tera === 'Stellar') {
			s = 1.2;
		}
		if ((aAb === 'protean' || aAb === 'libero') && !att.tera) s = Math.max(s, 1.5);
		return s;
	}

	function movePriority(ctx, att, mv) {
		var p = mv.priority || 0;
		if (!att.abilityKnown) return p;
		if (att.ability === 'prankster' && mv.category === 'Status') p += 1;
		if (att.ability === 'galewings' && mv.type === 'Flying' && (ctx.field.gen < 7 || att.hp >= att.maxhp)) p += 1;
		if (att.ability === 'triage' && (mv.drain || mv.flags.heal)) p += 3;
		if (mv.id === 'grassyglide' && ctx.field.terrain === 'grassy' && att.grounded) p += 1;
		return p;
	}

	/**
	 * Damage of one use of `mv` from `att` into `def`.
	 * Returns {min, max, avg, acc, exp, immune, immuneChance, hits, priority, type, category}
	 * where exp = average roll x accuracy x (1 - chance an unrevealed ability blocks it).
	 */
	function calc(ctx, att, def, mv, opt) {
		opt = opt || {};
		var f = ctx.field, gen = f.gen, battle = ctx.battle;
		var id = mv.id;
		var res = {min: 0, max: 0, avg: 0, acc: 1, exp: 0, immune: false, immuneChance: 0, hits: 1, priority: movePriority(ctx, att, mv), type: mv.type, category: mv.category, moldBreaker: false};
		if (!mv || mv.category === 'Status') return res;
		var zero = function () { res.immune = true; return res; };
		if (NO_DAMAGE_MOVES[id]) return zero();
		if (ONE_TIME_MOVES[id] && opt.firstTurn === false) return zero();
		if (TYPE_BURN_MOVES[id] && att.types.indexOf(TYPE_BURN_MOVES[id]) < 0) return zero();

		if (!att.abilityKnown && !opt._ab && att.possibleAbilities && att.possibleAbilities.length > 1 &&
			att.possibleAbilities.some(function (a) { return DAMAGE_ABILITIES[a]; })) {
			var oAb = clone(opt, {_ab: true});
			var parts = att.possibleAbilities.map(function (a) { return calc(ctx, clone(att, {ability: a, abilityKnown: true}), def, mv, oAb); });
			var n0 = parts.length;
			var mixed = clone(parts[0], {priority: res.priority, immune: parts.every(function (q) { return q.immune; })});
			['avg', 'exp', 'acc', 'immuneChance'].forEach(function (k0) {
				mixed[k0] = parts.reduce(function (x, q) { return x + q[k0]; }, 0) / n0;
			});
			// The range covers every ability it might have, so a hit that kills with one of them
			// shows up as a risk instead of being averaged away.
			if (cfg.wideRange) {
				mixed.min = Math.min.apply(null, parts.map(function (q) { return q.min; }));
				mixed.max = Math.max.apply(null, parts.map(function (q) { return q.max; }));
			} else {
				mixed.min = parts.reduce(function (x, q) { return x + q.min; }, 0) / n0;
				mixed.max = parts.reduce(function (x, q) { return x + q.max; }, 0) / n0;
			}
			return mixed;
		}

		var aAb = att.abilityKnown ? att.ability : '';
		var moldBreaker = aAb === 'moldbreaker' || aAb === 'teravolt' || aAb === 'turboblaze' ||
			id === 'sunsteelstrike' || id === 'moongeistbeam' || id === 'photongeyser';
		res.moldBreaker = moldBreaker;
		var dCh = function (ids) { return moldBreaker ? 0 : abChance(def, ids); };
		var blend = function (chance, mult) { return 1 + chance * (mult - 1); };

		var type = mv.type, cat = mv.category, bp = mv.basePower || 0;
		var atkNow = (att.stats.atk || 1) * boostMult(att.boosts.atk);
		var spaNow = (att.stats.spa || 1) * boostMult(att.boosts.spa);

		// ---- type and category changes
		var ateBoost = false;
		if (id === 'terablast' && att.tera) {
			if (att.tera !== 'Stellar') type = att.tera; else bp = 100;
			if (atkNow > spaNow) cat = 'Physical';
		}
		if (id === 'weatherball') {
			if (f.sun) { type = 'Fire'; bp = 100; } else if (f.rain) { type = 'Water'; bp = 100; } else if (f.sand) { type = 'Rock'; bp = 100; } else if (f.snow || f.hail) { type = 'Ice'; bp = 100; }
		}
		if (id === 'terrainpulse' && att.grounded && f.terrain) {
			type = {electric: 'Electric', grassy: 'Grass', psychic: 'Psychic', misty: 'Fairy'}[f.terrain];
			bp = 100;
		}
		// Judgment takes the type of the Plate and Multi-Attack the type of the Memory, which is the
		// forme's own type. Terastallizing does not change either. Revelation Dance follows the
		// user's current first type, Tera included.
		if (id === 'judgment') type = /plate$/.test(att.item || '') && TYPE_ITEMS[att.item] ? TYPE_ITEMS[att.item] : ((att.origTypes && att.origTypes[0]) || att.types[0] || type);
		if (id === 'multiattack') type = (att.origTypes && att.origTypes[0]) || att.types[0] || type;
		if (id === 'aurawheel' && /hangry/.test(att.forme || att.species.id || '')) type = 'Dark';
		if (id === 'revelationdance') type = att.types[0] || type;
		if (id === 'ragingbull') {
			var tid = att.species.id || '';
			if (tid.indexOf('combat') >= 0) type = 'Fighting'; else if (tid.indexOf('blaze') >= 0) type = 'Fire'; else if (tid.indexOf('aqua') >= 0) type = 'Water';
		}
		if (id === 'ivycudgel') {
			var oid = att.species.id || '';
			if (oid.indexOf('wellspring') >= 0) type = 'Water'; else if (oid.indexOf('hearthflame') >= 0) type = 'Fire'; else if (oid.indexOf('cornerstone') >= 0) type = 'Rock';
		}
		if (id === 'photongeyser' && atkNow > spaNow) cat = 'Physical';
		if (!mv.isMaxed && !mv.isZ) {
			if (type === 'Normal' && id !== 'terablast' && id !== 'weatherball') {
				var ate = {aerilate: 'Flying', pixilate: 'Fairy', refrigerate: 'Ice', galvanize: 'Electric'}[aAb];
				if (ate) { type = ate; ateBoost = true; }
			}
			if (aAb === 'normalize') { type = 'Normal'; ateBoost = gen >= 7; }
			if (aAb === 'liquidvoice' && mv.flags.sound) type = 'Water';
		}
		res.type = type;
		res.category = cat;

		// ---- type effectiveness
		var eff = 1;
		def.types.forEach(function (t) {
			var m = typeMult(battle, type, t);
			if (id === 'freezedry' && t === 'Water') m = 2;
			if (id === 'thousandarrows' && t === 'Flying') m = 1;
			if (m === 0) {
				if ((aAb === 'scrappy' || aAb === 'mindseye') && t === 'Ghost' && (type === 'Normal' || type === 'Fighting')) m = 1;
				if (def.item === 'ringtarget') m = 1;
				if (type === 'Ground' && t === 'Flying' && def.grounded) m = 1;
			}
			eff *= m;
		});
		if (id === 'flyingpress') def.types.forEach(function (t) { eff *= typeMult(battle, 'Flying', t); });
		if (type === 'Stellar') eff = def.tera ? 2 : 1;
		if (eff === 0) return zero();

		// ---- immunities from abilities, items and field
		var immuneChance = 0;
		if (type === 'Ground' && id !== 'thousandarrows' && !f.gravity && !mv.isFixed) {
			var dv = def.volatiles || {};
			if (!(def.item === 'ironball' || dv.smackdown || dv.ingrain)) {
				if (def.item === 'airballoon' || dv.magnetrise || dv.telekinesis) return zero();
				immuneChance = Math.max(immuneChance, dCh(['levitate']));
				if (!def.itemKnown && def.possibleItems.indexOf('airballoon') >= 0) immuneChance = Math.max(immuneChance, 0.5);
			}
		}
		immuneChance = Math.max(immuneChance, dCh(IMMUNE_ABILITIES[type] || []));
		if (mv.flags.sound) immuneChance = Math.max(immuneChance, dCh(['soundproof']));
		if (mv.flags.bullet) immuneChance = Math.max(immuneChance, dCh(['bulletproof']));
		if (mv.flags.wind) immuneChance = Math.max(immuneChance, dCh(['windrider']));
		if (eff <= 1) immuneChance = Math.max(immuneChance, dCh(['wonderguard']));
		if (res.priority > 0) {
			if (f.terrain === 'psychic' && def.grounded) return zero();
			immuneChance = Math.max(immuneChance, dCh(['queenlymajesty', 'dazzling', 'armortail']));
		}
		if (f.weather === 'primordialsea' && type === 'Fire') return zero();
		if (f.weather === 'desolateland' && type === 'Water') return zero();
		if (immuneChance >= 0.999) return zero();
		res.immuneChance = immuneChance;

		var finish = function (lo, hi, acc, critAvg, avg) {
			res.min = Math.max(0, lo);
			res.max = Math.max(0, hi);
			res.avg = (avg === undefined ? (res.min + res.max) / 2 : avg) * (critAvg || 1);
			res.acc = clamp(acc, 0, 1);
			res.exp = res.avg * res.acc * (1 - immuneChance);
			return res;
		};

		// ---- accuracy
		var acc = mv.accuracy === true ? 1 : (mv.accuracy || 100) / 100;
		if (aAb === 'noguard' || dCh(['noguard']) >= 1) {
			acc = 1;
		} else if (mv.accuracy !== true) {
			if (id === 'thunder' || id === 'hurricane') { if (f.rain) acc = 1; else if (f.sun) acc = 0.5; }
			if (id === 'blizzard' && (f.snow || f.hail)) acc = 1;
			if (aAb === 'compoundeyes') acc *= 1.3;
			if (aAb === 'hustle' && cat === 'Physical') acc *= 0.8;
			if (aAb === 'victorystar') acc *= 1.1;
			if (att.item === 'widelens') acc *= 1.1;
			var evade = aAb === 'unaware' || aAb === 'keeneye' || aAb === 'mindseye' ? 0 : (def.boosts.evasion || 0);
			acc *= accMult((att.boosts.accuracy || 0) - evade);
			if (def.item === 'brightpowder') acc *= 0.9;
			if (gen === 1) acc *= 255 / 256;
		}
		if (mv.ohko) {
			if (att.level < def.level || dCh(['sturdy']) >= 1) return zero();
			if (id === 'sheercold' && def.types.indexOf('Ice') >= 0 && gen >= 7) return zero();
			return finish(def.hp, def.hp, 0.3, 1);
		}

		// ---- fixed damage
		if (mv.damage === 'level' || id === 'seismictoss' || id === 'nightshade' || id === 'psywave') return finish(att.level, att.level, acc, 1);
		if (typeof mv.damage === 'number') return finish(mv.damage, mv.damage, acc, 1);
		if (id === 'superfang' || id === 'ruination' || id === 'naturesmadness') { var half = Math.max(1, Math.floor(def.hp / 2)); return finish(half, half, acc, 1); }
		if (id === 'endeavor') { var ed = Math.max(0, def.hp - att.hp); return finish(ed, ed, acc, 1); }
		if (id === 'finalgambit') return finish(att.hp, att.hp, acc, 1);

		// ---- base power
		var aSpe = effSpeed(ctx, att), dSpe = effSpeed(ctx, def);
		var hits = 1;
		if (!mv.isMaxed && !mv.isZ) {
			if (id === 'lowkick' || id === 'grassknot') bp = weightPower(def.weightkg);
			if (id === 'heavyslam' || id === 'heatcrash') { var wr = att.weightkg / Math.max(0.1, def.weightkg); bp = wr >= 5 ? 120 : wr >= 4 ? 100 : wr >= 3 ? 80 : wr >= 2 ? 60 : 40; }
			if (id === 'gyroball') bp = Math.min(150, Math.floor(25 * dSpe / Math.max(1, aSpe)) + 1);
			if (id === 'electroball') { var sr = aSpe / Math.max(1, dSpe); bp = sr >= 4 ? 150 : sr >= 3 ? 120 : sr >= 2 ? 80 : sr >= 1 ? 60 : 40; }
			if (id === 'reversal' || id === 'flail') { var hr = 48 * att.hp / att.maxhp; bp = hr < 2 ? 200 : hr < 5 ? 150 : hr < 10 ? 100 : hr < 17 ? 80 : hr < 33 ? 40 : 20; }
			if (id === 'eruption' || id === 'waterspout' || id === 'dragonenergy') bp = Math.max(1, Math.floor(150 * att.hp / att.maxhp));
			if (id === 'storedpower' || id === 'powertrip') { var sum = 0; for (var bk in att.boosts) if (att.boosts[bk] > 0) sum += att.boosts[bk]; bp = 20 + 20 * sum; }
			if (id === 'crushgrip' || id === 'wringout') bp = Math.max(1, Math.floor(120 * def.hp / def.maxhp));
			if (id === 'hardpress') bp = Math.max(1, Math.floor(100 * def.hp / def.maxhp));
			if (id === 'return' || id === 'frustration') bp = 102;
			if (id === 'punishment') { var ps = 0; for (var pk in def.boosts) if (def.boosts[pk] > 0) ps += def.boosts[pk]; bp = Math.min(200, 60 + 20 * ps); }
			if (id === 'magnitude') bp = 71;
			if (id === 'present') bp = 40;
			if (id === 'trumpcard') bp = 40;
			if (id === 'beatup') bp = 30;
			if (id === 'ragefist') bp = Math.min(350, 50 + 50 * (att.timesAttacked || 0));
			if (id === 'lastrespects') bp = Math.min(150, 50 + 50 * (opt.fallen || 0));
			if (id === 'facade' && att.status && att.status !== 'slp') bp *= 2;
			if ((id === 'hex' || id === 'infernalparade') && def.status) bp *= 2;
			if ((id === 'venoshock' || id === 'barbbarrage') && (def.status === 'psn' || def.status === 'tox')) bp *= 2;
			if (id === 'brine' && def.hp * 2 <= def.maxhp) bp *= 2;
			if (id === 'acrobatics' && !att.item && att.itemKnown) bp *= 2;
			if (id === 'knockoff' && gen >= 6 && (def.item || !def.itemKnown)) bp *= 1.5;
			if (id === 'poltergeist' && def.itemKnown && !def.item) return zero();
			if ((id === 'solarbeam' || id === 'solarblade') && (f.rain || f.sand || f.snow || f.hail)) bp *= 0.5;
			if ((id === 'boltbeak' || id === 'fishiousrend') && opt.attFirst !== false) bp *= 2;
			if ((id === 'payback' || id === 'avalanche' || id === 'revenge') && opt.attFirst === false) bp *= 2;
			if (id === 'expandingforce' && f.terrain === 'psychic' && att.grounded) bp *= 1.5;
			if (id === 'risingvoltage' && f.terrain === 'electric' && def.grounded) bp *= 2;
			if (id === 'psyblade' && f.terrain === 'electric') bp *= 1.5;
			if (id === 'wakeupslap' && def.status === 'slp') bp *= 2;
			if (id === 'smellingsalts' && def.status === 'par') bp *= 2;
			if (id === 'tripleaxel') { bp = aAb === 'skilllink' ? 120 : 94; acc = 1; }
			if (id === 'triplekick') { bp = aAb === 'skilllink' ? 60 : 47; acc = 1; }
			if (id === 'populationbomb') { hits = aAb === 'skilllink' ? 10 : att.item === 'widelens' ? 9.5 : att.item === 'loadeddice' ? 7 : 5.9; acc = 1; } else if (id !== 'tripleaxel' && id !== 'triplekick' && mv.multihit) {
				if (Array.isArray(mv.multihit)) {
					var lo = mv.multihit[0], hi = mv.multihit[1];
					hits = aAb === 'skilllink' ? hi : att.item === 'loadeddice' ? (hi === 5 ? 4.5 : hi) : (lo === 2 && hi === 5 ? 3.1 : (lo + hi) / 2);
				} else {
					hits = mv.multihit;
				}
			}
		}
		if (!bp) return zero();
		res.hits = hits;

		// ---- attacking and defending stats
		var aStat = cat === 'Physical' ? 'atk' : 'spa';
		var dStat = cat === 'Physical' ? 'def' : 'spd';
		if (!mv.isMaxed && !mv.isZ) {
			if (mv.overrideOffensiveStat) aStat = mv.overrideOffensiveStat;
			if (mv.overrideDefensiveStat) dStat = mv.overrideDefensiveStat;
		}
		var aSrc = mv.overrideOffensivePokemon === 'target' ? def : att;
		var crit = !!mv.willCrit && dCh(['battlearmor', 'shellarmor']) < 1;
		var aBoost = aSrc.boosts[aStat] || 0;
		var dBoost = def.boosts[dStat] || 0;
		if (crit) { aBoost = Math.max(0, aBoost); dBoost = Math.min(0, dBoost); }
		if (aAb === 'unaware' || mv.ignoreDefensive) dBoost = 0;
		var unaware = dCh(['unaware']);
		var A = (aSrc.stats[aStat] || 1) * (unaware * 1 + (1 - unaware) * boostMult(aBoost));
		var D = (def.stats[dStat] || 1) * boostMult(dBoost);
		var av = att.volatiles || {}, dvv = def.volatiles || {};

		if (aStat === 'atk' && aSrc === att) {
			if (aAb === 'hugepower' || aAb === 'purepower') A *= 2;
			if (aAb === 'guts' && att.status) A *= 1.5;
			if (aAb === 'toxicboost' && (att.status === 'psn' || att.status === 'tox')) A *= 1.5;
			if (aAb === 'hustle' || aAb === 'gorillatactics') A *= 1.5;
			if (aAb === 'orichalcumpulse' && f.sun) A *= 4 / 3;
			if (att.item === 'choiceband') A *= 1.5;
			if (att.item === 'thickclub' && /^(cubone|marowak)/.test(att.species.id || '')) A *= 2;
			if (av.protosynthesisatk || av.quarkdriveatk) A *= 1.3;
		} else if (aStat === 'spa') {
			if (aAb === 'solarpower' && f.sun) A *= 1.5;
			if (aAb === 'flareboost' && att.status === 'brn') A *= 1.5;
			if (aAb === 'hadronengine' && f.terrain === 'electric') A *= 4 / 3;
			if (att.item === 'choicespecs') A *= 1.5;
			if (av.protosynthesisspa || av.quarkdrivespa) A *= 1.3;
		} else if (aStat === 'def') {
			if (av.protosynthesisdef || av.quarkdrivedef) A *= 1.3;
		}
		if (att.item === 'lightball' && (att.species.id || '').indexOf('pikachu') === 0) A *= 2;
		if (att.hp * 3 <= att.maxhp && ((aAb === 'blaze' && type === 'Fire') || (aAb === 'torrent' && type === 'Water') || (aAb === 'overgrow' && type === 'Grass') || (aAb === 'swarm' && type === 'Bug'))) A *= 1.5;
		if (type === 'Fire' || type === 'Ice') A *= blend(dCh(['thickfat']), 0.5);

		if (dStat === 'spd') {
			if (f.sand && def.types.indexOf('Rock') >= 0 && gen >= 4) D *= 1.5;
			if (def.item === 'assaultvest') D *= 1.5;
			else if (!def.itemKnown && def.itemW && def.itemW.assaultvest) D *= 1 + 0.5 * def.itemW.assaultvest;
			if (dvv.protosynthesisspd || dvv.quarkdrivespd) D *= 1.3;
		} else {
			if (f.snow && def.types.indexOf('Ice') >= 0) D *= 1.5;
			D *= blend(dCh(['furcoat']), 2);
			if (def.status) D *= blend(dCh(['marvelscale']), 1.5);
			if (dvv.protosynthesisdef || dvv.quarkdrivedef) D *= 1.3;
		}
		var nfe = def.species.evos && def.species.evos.length;
		if (nfe && (def.item === 'eviolite' || (!def.itemKnown && def.possibleItems.indexOf('eviolite') >= 0))) D *= 1.5;

		// The four Ruin abilities cut one stat of everything else on the field to three quarters.
		if (aSrc === att) {
			if (aStat === 'atk' && aAb !== 'tabletsofruin') A *= blend(abChance(def, ['tabletsofruin']), 0.75);
			if (aStat === 'spa' && aAb !== 'vesselofruin') A *= blend(abChance(def, ['vesselofruin']), 0.75);
			// Regigigas hits at half strength for its first five turns out.
			if (aStat === 'atk' && aAb === 'slowstart' && (av.slowstart || att.benched)) A *= 0.5;
		}
		if (dStat === 'def' && abChance(def, ['swordofruin']) < 1) D *= blend(abChance(att, ['swordofruin']), 0.75);
		if (dStat === 'spd' && abChance(def, ['beadsofruin']) < 1) D *= blend(abChance(att, ['beadsofruin']), 0.75);

		var base = Math.floor(Math.floor(Math.floor(2 * att.level / 5 + 2) * bp * A / Math.max(1, D)) / 50) + 2;

		// ---- modifiers
		var mod = 1;
		if (av.charge && type === 'Electric') mod *= 2;
		if (f.sun) { if (type === 'Fire') mod *= 1.5; else if (type === 'Water') mod *= id === 'hydrosteam' ? 1.5 : 0.5; }
		if (f.rain) { if (type === 'Water') mod *= 1.5; else if (type === 'Fire') mod *= 0.5; }
		if (crit) mod *= (gen >= 6 ? 1.5 : 2) * (aAb === 'sniper' ? 1.5 : 1);
		mod *= stabMult(att, type, aAb);
		mod *= eff;
		if (att.status === 'brn' && cat === 'Physical' && aAb !== 'guts' && id !== 'facade') mod *= 0.5;
		if (!crit && aAb !== 'infiltrator' && id !== 'brickbreak' && id !== 'psychicfangs' && id !== 'ragingbull') {
			var sc = sideOf(ctx, def) || {};
			if (sc.auroraveil || (cat === 'Physical' && sc.reflect) || (cat === 'Special' && sc.lightscreen)) mod *= 0.5;
		}
		if (aAb === 'technician' && bp <= 60) mod *= 1.5;
		if (aAb === 'toughclaws' && mv.flags.contact) mod *= 1.3;
		if (aAb === 'sheerforce' && (mv.secondary || mv.secondaries)) mod *= 1.3;
		if (aAb === 'ironfist' && mv.flags.punch) mod *= 1.2;
		if (aAb === 'strongjaw' && mv.flags.bite) mod *= 1.5;
		if (aAb === 'megalauncher' && mv.flags.pulse) mod *= 1.5;
		if (aAb === 'sharpness' && mv.flags.slicing) mod *= 1.5;
		if (aAb === 'reckless' && (mv.recoil || mv.hasCrashDamage)) mod *= 1.2;
		if (aAb === 'punkrock' && mv.flags.sound) mod *= 1.3;
		if (aAb === 'tintedlens' && eff < 1) mod *= 2;
		if (aAb === 'neuroforce' && eff > 1) mod *= 1.25;
		if (aAb === 'waterbubble' && type === 'Water') mod *= 2;
		if ((aAb === 'steelworker' || aAb === 'steelyspirit') && type === 'Steel') mod *= 1.5;
		if (aAb === 'transistor' && type === 'Electric') mod *= gen >= 9 ? 1.3 : 1.5;
		if (aAb === 'dragonsmaw' && type === 'Dragon') mod *= 1.5;
		if (aAb === 'rockypayload' && type === 'Rock') mod *= 1.5;
		if (aAb === 'sandforce' && f.sand && (type === 'Rock' || type === 'Ground' || type === 'Steel')) mod *= 1.3;
		if (aAb === 'analytic' && opt.attFirst === false) mod *= 1.3;
		if (aAb === 'parentalbond' && hits === 1) mod *= 1.25;
		if (ateBoost) mod *= gen >= 7 ? 1.2 : 1.3;
		if (aAb === 'supremeoverlord') {
			var fallen = 0;
			for (var n = 1; n <= 5; n++) if (av['fallen' + n]) fallen = n;
			mod *= 1 + 0.1 * fallen;
		}
		if (def.hp >= def.maxhp) mod *= blend(dCh(['multiscale', 'shadowshield']), 0.5);
		if (eff > 1) mod *= blend(dCh(['filter', 'solidrock', 'prismarmor']), 0.75);
		if (cat === 'Special') mod *= blend(dCh(['icescales']), 0.5);
		if (mv.flags.contact) mod *= blend(dCh(['fluffy']), 0.5);
		if (type === 'Fire') mod *= blend(dCh(['fluffy']), 2) * blend(dCh(['heatproof', 'waterbubble']), 0.5) * blend(dCh(['dryskin']), 1.25);
		if (type === 'Ghost') mod *= blend(dCh(['purifyingsalt']), 0.5);
		if (mv.flags.sound) mod *= blend(dCh(['punkrock']), 0.5);
		// An item I have not seen yet: the average over the ones its sets carry, and the range
		// from the weakest to the strongest of them.
		var iLo = 1, iHi = 1, iAvg = 1;
		if (!att.itemKnown && att.itemW) {
			var iSum = 0, iTot = 0, iMin = Infinity, iMax = 0;
			for (var iw in att.itemW) {
				var ip = itemPower(iw, cat, type, eff, att);
				iTot += att.itemW[iw];
				iSum += att.itemW[iw] * ip;
				iMin = Math.min(iMin, ip);
				iMax = Math.max(iMax, ip);
			}
			if (iTot > 0) { iAvg = iSum / iTot; iLo = iMin; iHi = iMax; }
			if (!cfg.wideRange) { iLo = iAvg; iHi = iAvg; }
		}
		if (att.item === 'lifeorb') mod *= 1.3;
		if (att.item === 'expertbelt' && eff > 1) mod *= 1.2;
		if (TYPE_ITEMS[att.item] === type) mod *= 1.2;
		if (att.item === 'muscleband' && cat === 'Physical') mod *= 1.1;
		if (att.item === 'wiseglasses' && cat === 'Special') mod *= 1.1;
		if (att.item === 'punchingglove' && mv.flags.punch) mod *= 1.1;
		var tb = gen >= 8 ? 1.3 : 1.5;
		if (att.grounded) {
			if (f.terrain === 'electric' && type === 'Electric') mod *= tb;
			if (f.terrain === 'grassy' && type === 'Grass') mod *= tb;
			if (f.terrain === 'psychic' && type === 'Psychic') mod *= tb;
		}
		if (def.grounded) {
			if (f.terrain === 'misty' && type === 'Dragon') mod *= 0.5;
			if (f.terrain === 'grassy' && (id === 'earthquake' || id === 'bulldoze' || id === 'magnitude')) mod *= 0.5;
		}
		if ((id === 'collisioncourse' || id === 'electrodrift') && eff > 1) mod *= 4 / 3;

		var one = base * mod;
		// Triple Axel and Population Bomb stop at the first miss, so the total swings widely.
		var sLo = 1, sHi = 1;
		if (!mv.isMaxed && !mv.isZ && aAb !== 'skilllink') {
			if (id === 'tripleaxel') { sLo = 20 / 94; sHi = 120 / 94; }
			if (id === 'triplekick') { sLo = 10 / 47; sHi = 60 / 47; }
			if (id === 'populationbomb') { sLo = 0.35; sHi = 10 / hits; }
		}
		var hi2 = Math.floor(one * iHi * sHi) * hits;
		var lo2 = Math.floor(one * iLo * sLo * 0.85) * hits;
		var avg2 = one * iAvg * 0.925 * hits;

		// Mimikyu's Disguise soaks the first hit.
		if (!opt.skipShield && !def.shieldDown && (def.species.id === 'mimikyu' || def.species.id === 'mimikyutotem') && dCh(['disguise']) > 0 && !dvv.disguisebroken) {
			var dd = gen >= 8 ? Math.floor(def.maxhp / 8) : 0;
			return finish(dd, dd, acc, 1);
		}
		// Eiscue's Ice Face soaks the first physical hit.
		if (!opt.skipShield && !def.shieldDown && def.species.id === 'eiscue' && dCh(['iceface']) > 0 && cat === 'Physical') return finish(0, 0, acc, 1);

		// Gen 1 critical hits depend on base Speed and happen often enough to matter.
		var critAvg = 1;
		if (gen === 1) {
			var bsSpe = (att.species.baseStats && att.species.baseStats.spe) || 80;
			var highCrit = id === 'slash' || id === 'razorleaf' || id === 'crabhammer' || id === 'karatechop';
			var rate = Math.min(0.996, bsSpe / 512 * (highCrit ? 8 : 1));
			critAvg = 1 + rate * ((2 * att.level + 5) / (att.level + 5) - 1);
		}
		return finish(lo2, hi2, acc, critAvg, avg2);
	}

	// ------------------------------------------------------------------ turn-level helpers

	// Chance the Pokemon actually gets to act when it tries to move.
	function actChance(ctx, c) {
		var p = 1;
		var v = c.volatiles || {};
		if (v.mustrecharge) return 0;
		if (c.status === 'slp') p *= ctx.field.gen === 1 ? 0.15 : 0.33;
		if (c.status === 'frz') p *= 0.2;
		if (c.status === 'par') p *= 0.75;
		if (v.confusion) p *= ctx.field.gen >= 7 ? 0.67 : 0.5;
		if (v.attract) p *= 0.5;
		return p;
	}

	// HP lost per turn to status, weather and so on. Negative means healing.
	function residual(ctx, c) {
		var f = ctx.field, gen = f.gen, m = c.maxhp, d = 0;
		var v = c.volatiles || {};
		var guard = hasAb(c, 'magicguard');
		if (!guard) {
			if (c.status === 'brn') d += m / (gen >= 7 ? 16 : 8);
			if (c.status === 'psn') d += hasAb(c, 'poisonheal') ? -m / 8 : m / (gen === 1 ? 16 : 8);
			if (c.status === 'tox') d += hasAb(c, 'poisonheal') ? -m / 8 : m * Math.min(15, (c.toxicTurns || 0) + 2) / 16;
			if (v.leechseed) d += m / (gen === 1 ? 16 : 8);
			if (v.saltcure) d += m / (c.types.indexOf('Water') >= 0 || c.types.indexOf('Steel') >= 0 ? 4 : 8);
			if (v.curse) d += m / 4;
			if (v.partiallytrapped) d += m / 8;
			if (f.sand && !['Rock', 'Ground', 'Steel'].some(function (t) { return c.types.indexOf(t) >= 0; }) && !hasAb(c, 'sandrush', 'sandforce', 'sandveil', 'overcoat') && c.item !== 'safetygoggles') d += m / 16;
			if (f.hail && c.types.indexOf('Ice') < 0 && !hasAb(c, 'icebody', 'snowcloak', 'overcoat', 'slushrush') && c.item !== 'safetygoggles') d += m / 16;
		}
		if (c.item === 'leftovers') d -= m / 16;
		if (c.item === 'blacksludge') d += c.types.indexOf('Poison') >= 0 ? -m / 16 : (guard ? 0 : m / 8);
		if (f.terrain === 'grassy' && c.grounded) d -= m / 16;
		if (v.aquaring || v.ingrain) d -= m / 16;
		return d;
	}

	function hazardDmg(ctx, c) {
		var sc = sideOf(ctx, c);
		if (!sc || c.item === 'heavydutyboots' || hasAb(c, 'magicguard')) return 0;
		var d = 0;
		if (sc.stealthrock) {
			var e = 1;
			c.types.forEach(function (t) { e *= typeMult(ctx.battle, 'Rock', t); });
			d += c.maxhp * e / 8;
		}
		if (sc.spikes && isGrounded(ctx.field, c)) {
			var n = clamp((sc.spikes[1] || 1), 1, 3);
			d += c.maxhp * [0, 1 / 8, 1 / 6, 1 / 4][n];
		}
		return Math.floor(d);
	}

	function lockedInto(c) {
		if (c.forcedMove) return c.forcedMove;
		if (!c.lastMove) return null;
		var v = c.volatiles || {};
		if (v.encore) return c.lastMove;
		if (c.itemKnown && /^choice(band|specs|scarf)$/.test(c.item)) return c.lastMove;
		return null;
	}
	// The same for an item I have not seen: how likely it is a Choice item holding it to the one
	// move it has used since it came in.
	function lockChance(c) {
		if (lockedInto(c)) return 1;
		if (c.mine || !c.lastMove || c.itemKnown || !(c.choiceP > 0)) return 0;
		if (!c.sinceIn || c.sinceIn.length !== 1 || c.sinceIn[0] !== c.lastMove) return 0;
		return clamp(c.choiceP, 0, 1);
	}

	// Moves the opponent might use: revealed ones for certain, then the rest of its possible set.
	function oppMoveList(ctx, opp) {
		var out = [];
		var v = opp.volatiles || {};
		if (v.mustrecharge) return out;
		var wrap = function (mv) { return opp.dynamaxed ? asMax(mv) : mv; };
		var locked = lockedInto(opp);
		if (locked) {
			out.push({mv: wrap(getMove(ctx.battle, locked)), w: 1});
			return out;
		}
		var free = 1 - lockChance(opp);
		opp.moves.forEach(function (id) { out.push({mv: wrap(getMove(ctx.battle, id)), w: id === opp.lastMove ? 1 : free}); });
		var wPool = (opp.moves.length >= 3 ? 0.5 : 0.75) * free;
		(opp.pool || []).forEach(function (id) { out.push({mv: wrap(getMove(ctx.battle, id)), w: wPool}); });
		if (!opp.hasSet && opp.moves.length < 4) {
			var cat = (opp.stats.atk || 1) * boostMult(opp.boosts.atk) >= (opp.stats.spa || 1) * boostMult(opp.boosts.spa) ? 'Physical' : 'Special';
			var ts = opp.origTypes.slice();
			if (opp.tera && opp.tera !== 'Stellar' && ts.indexOf(opp.tera) < 0) ts.push(opp.tera);
			ts.forEach(function (t) {
				out.push({mv: wrap({id: 'stab' + toID(t), name: t + ' STAB', type: t, basePower: 85, category: cat, accuracy: 100, priority: 0, flags: {}}), w: 0.7});
			});
		}
		return out;
	}

	function threat(ctx, opp, me, o) {
		var res = {exp: 0, best: null, items: []};
		var mult = actChance(ctx, opp);
		var top = 0;
		oppMoveList(ctx, opp).forEach(function (e) {
			if (e.mv.category === 'Status') return;
			var r = calc(ctx, opp, me, e.mv, o || {});
			// Explosion and the like cost them the Pokemon: a risk to keep in mind, not the move
			// to plan the whole matchup around.
			var boom = e.mv.selfdestruct || e.mv.id === 'finalgambit' ? 0.2 : 1;
			var w = e.w * mult * boom;
			res.items.push({mv: e.mv, r: r, w: w, p: e.w * boom});
			if (r.exp * w > top) { top = r.exp * w; res.best = {mv: e.mv, r: r, w: w}; }
		});
		// Expected damage of its best available move: the strongest one if it has it, otherwise
		// the next strongest, and so on down the list.
		var left = 1;
		res.items.slice().sort(function (a, b) { return b.r.exp - a.r.exp; }).forEach(function (it) {
			var pHas = Math.min(1, it.p);
			res.exp += left * pHas * it.r.exp;
			left *= 1 - pHas;
		});
		res.exp *= mult;
		return res;
	}

	// Moves and abilities that undo a boost before it pays off.
	var BOOST_ANSWERS = {haze: 1, clearsmog: 1, roar: 1, whirlwind: 1, dragontail: 1, circlethrow: 1, encore: 1, topsyturvy: 1, spectralthief: 1, psychup: 1, yawn: 1, perishsong: 1};

	// What a boost is worth beyond the Pokemon in front of me: how much better each of the
	// opponent's other Pokemon goes for me once I have it. This is the reason to set up at all,
	// since against the Pokemon already on the field the turn spent boosting rarely pays.
	function sweepGain(S, me, me2, hp) {
		if (!(cfg.sweepWeight > 0) || hp <= 0) return 0;
		var ctx = S.ctx;
		var now = withHp(me, hp), up = withHp(me2, hp);
		var gain = function (Y, hpY) {
			var a = race(ctx, now, Y, hp, hpY, {ignoreLock: true}).score;
			var b = race(ctx, up, Y, hp, hpY, {ignoreLock: true}).score;
			return Math.max(0, b - a);
		};
		var gains = (S.benchOpp || []).map(function (Y) { return gain(Y, Y.hp); });
		var unseen = Math.max(0, S.oppAlive - 1 - gains.length);
		if (unseen > 0) {
			// The ones not revealed yet: assume they go like the known ones, or like a fresh
			// copy of the one in front of me when nothing else is known.
			var typical = gains.length ? gains.reduce(function (x, y) { return x + y; }, 0) / gains.length : gain(S.opp, S.opp.maxhp);
			for (var i = 0; i < unseen; i++) gains.push(typical * 0.7);
		}
		gains.sort(function (x, y) { return y - x; });
		// Each later fight starts from less HP and less certainty, so it counts for less.
		var w = [1, 0.6, 0.35, 0.2, 0.1], total = 0;
		gains.forEach(function (g, j) { total += g * (w[j] || 0); });
		// It only pays if I am still standing after the Pokemon in front of me.
		var here = race(ctx, up, S.opp, hp, S.opp.hp);
		total *= 0.25 + 0.75 * (here.win === undefined ? 0.5 : here.win);
		// Something on their side that removes boosts makes the whole plan unreliable.
		var answered = (S.opp.moves || []).some(function (m) { return BOOST_ANSWERS[m]; }) || abChance(S.opp, ['unaware']) >= 0.5;
		(S.benchOpp || []).forEach(function (Y) {
			if ((Y.moves || []).some(function (m) { return BOOST_ANSWERS[m]; }) || (Y.abilityKnown && Y.ability === 'unaware')) answered = true;
		});
		if (answered) total *= 0.3;
		return cfg.sweepWeight * total;
	}

	// The stat drop an attack inflicts on its target, when it is likely enough to plan around.
	function targetDrop(S, mv) {
		if (!cfg.dropAware) return null;
		var me = S.me, opp = S.opp;
		if (hasAb(me, 'sheerforce') || (opp.volatiles || {}).substitute) return null;
		var secs = mv.secondaries || (mv.secondary ? [mv.secondary] : []);
		var out = null;
		secs.forEach(function (sec) {
			if (!sec || !sec.boosts) return;
			var ch = Math.min(1, (sec.chance || 100) / 100 * (hasAb(me, 'serenegrace') ? 2 : 1));
			if (ch < 0.5) return;
			var down = false;
			for (var k in sec.boosts) if (sec.boosts[k] < 0) down = true;
			if (down) out = {boosts: sec.boosts, ch: ch};
		});
		if (!out) return null;
		// Abilities and items that stop the drop, turn it around, or punish it.
		var blocked = abChance(opp, ['clearbody', 'whitesmoke', 'fullmetalbody', 'shielddust', 'mirrorarmor']);
		if (opp.item === 'clearamulet' || opp.item === 'covertcloak') blocked = 1;
		out.ch *= 1 - blocked;
		if (out.ch < 0.3 || abChance(opp, ['contrary', 'defiant', 'competitive']) >= 0.5) return null;
		var b2 = Object.assign({}, opp.boosts), changed = false;
		for (var k2 in out.boosts) {
			var v = clamp((b2[k2] || 0) + out.boosts[k2], -6, 6);
			if (v !== (b2[k2] || 0)) changed = true;
			b2[k2] = v;
		}
		if (!changed) return null;
		out.opp = clone(opp, {boosts: b2});
		return out;
	}

	// How much more a boosted Pokemon is worth knocking out than its HP alone says.
	function boostWorth(opp) {
		if (!(cfg.boostWorth > 0)) return 0;
		var b = opp.boosts || {};
		var n = Math.max(0, b.atk || 0) + Math.max(0, b.spa || 0) + Math.max(0, b.spe || 0) + 0.4 * (Math.max(0, b.def || 0) + Math.max(0, b.spd || 0));
		return Math.min(0.4, cfg.boostWorth * n);
	}

	// The opponent picks its move for the typing it can see. If I Terastallize this turn, that
	// move lands on the new typing instead. This is where a defensive Tera wins the exchange:
	// the super-effective hit they clicked becomes a resisted one.
	function surpriseThreat(ctx, opp, meNow, meT) {
		var seen = threat(ctx, opp, meNow);
		var after = threat(ctx, opp, meT);
		if (!seen.best) return after;
		// How sure I am they click what looks best right now. A locked move leaves no doubt.
		var trust = lockedInto(opp) ? 1 : 0.7;
		var r = calc(ctx, opp, meT, seen.best.mv, {});
		var w = seen.best.w;
		var res = {exp: trust * r.exp * w + (1 - trust) * after.exp, best: {mv: seen.best.mv, r: r, w: w}, items: [{mv: seen.best.mv, r: r, w: w}], surprise: true};
		after.items.forEach(function (it) { res.items.push({mv: it.mv, r: it.r, w: it.w * (1 - trust)}); });
		return res;
	}

	// Whether the opponent still has its Tera, what it could become, and how tempted it is.
	function oppTeraGuess(ctx, me, opp) {
		if (!(cfg.predictTera > 0) || ctx.gen !== 9 || opp.tera || !opp.possTera || !opp.possTera.length) return null;
		var far = ctx.battle.farSide;
		if (!far || (far.pokemon || []).some(function (q) { return q.terastallized; })) return null;
		var kinds = opp.possTera.filter(function (t) { return t && t !== 'Stellar'; }).slice(0, 4);
		if (!kinds.length) return null;
		var forms = kinds.map(function (t) {
			var o = clone(opp, {tera: t, types: [t]});
			o.grounded = isGrounded(ctx.field, o);
			return o;
		});
		// Good players hold their Tera for the second half of the game.
		var theirLost = (far.pokemon || []).filter(function (q) { return q.fainted; }).length;
		var down = theirLost + ((ctx.req.side && ctx.req.side.pokemon) || []).filter(function (q) { return parseCond(q.condition).fainted; }).length;
		var stage = 0.5 + 1.5 * clamp(down / 4, 0, 1);
		var p = cfg.predictTera * stage;
		var atk = bestAttack(ctx, me, opp, {});
		var blunted = false, saved = false;
		if (atk.mv && atk.exp >= 0.4 * opp.maxhp) {
			// I am threatening real damage. If one of those types blunts it, a Tera is their way out.
			blunted = forms.some(function (o) { return calc(ctx, me, o, atk.mv, {firstTurn: false}).exp <= 0.6 * atk.exp; });
			if (blunted) p = Math.max(p, cfg.predictTeraPickle * stage);
			// Most of all when it turns a KO into a hit the Pokemon survives.
			saved = !!atk.r && koProb(atk.r, opp.hp, opp) >= 0.7 && forms.some(function (o) { return koProb(calc(ctx, me, o, atk.mv, {firstTurn: false}), o.hp, o) <= 0.2; });
			if (saved) p = Math.max(p, 0.2 * stage);
		}
		if (cfg.teraCal) {
			// What opponents did over 4276 turns of ladder games in which they still had it: 3.8% a
			// turn overall. With nothing at stake 0.6% (none of theirs lost) to 6% (four lost);
			// when a Tera blunts my hit 1.8% to 19%; when it saves the Pokemon from a KO 3.8% to 22%.
			// A boosted Pokemon uses it more, and hardly anyone does in the first two turns.
			var rise = [1, 1.9, 2.5, 5.5, 7.5, 7.5][Math.min(5, theirLost)];
			p = (saved ? 0.04 : blunted ? 0.02 : 0.006) * rise;
			var up = 0;
			for (var bk in opp.boosts) if (opp.boosts[bk] > 0) up++;
			if (up) p *= 1.6;
			if (ctx.battle.turn <= 2) p *= 0.25;
			return {p: clamp(p, 0, 0.3), forms: forms};
		}
		return {p: clamp(p, 0, 0.5), forms: forms};
	}

	function koProb(r, hp, def) {
		if (!r || r.immune || r.max <= 0) return 0;
		var p;
		if (r.min >= hp) p = 1; else if (r.max < hp) p = 0; else p = (r.max - hp + 1) / (r.max - r.min + 1);
		p *= r.acc * (1 - (r.immuneChance || 0));
		if (p > 0 && def.hp >= def.maxhp && r.hits <= 1) {
			var block = 0;
			if (def.itemKnown) { if (def.item === 'focussash') block = 1; } else if (def.possibleItems.indexOf('focussash') >= 0) block = 0.5;
			if (!r.moldBreaker) block = Math.max(block, abChance(def, ['sturdy']));
			p *= 1 - block;
		}
		return p;
	}

	// Probability that I move before the opponent.
	function firstProb(ctx, me, opp, myPrio, oppPrio) {
		if (myPrio !== oppPrio) return myPrio > oppPrio ? 1 : 0;
		var sMe = effSpeed(ctx, me), sOpp = effSpeed(ctx, opp);
		if (sMe === sOpp) return 0.5;
		var faster = sMe > sOpp;
		var p = (ctx.field.trickRoom ? !faster : faster) ? 1 : 0;
		var ratio = sMe / sOpp;
		if (!ctx.field.trickRoom) {
			if (p === 1 && !opp.itemKnown && opp.possibleItems.indexOf('choicescarf') >= 0 && ratio < 1.5) p = 0.6;
			if (ratio > 0.96 && ratio < 1.04) p = 0.5 + (p - 0.5) * 0.5;
		}
		return p;
	}

	// The strongest attack `me` can keep using, for projecting how a matchup plays out.
	function bestAttack(ctx, me, opp, opt) {
		opt = opt || {};
		var out = {exp: 0, r: null, mv: null, priority: 0, prio: null};
		var ids = opt.moves || me.moves || [];
		var locked = me.mine && me.itemKnown && /^choice/.test(me.item) && me.lastMove && ids.indexOf(me.lastMove) >= 0 && !opt.ignoreLock ? me.lastMove : null;
		ids.forEach(function (id) {
			if (locked && id !== locked) return;
			var mv = getMove(ctx.battle, id);
			if (me.dynamaxed) mv = asMax(mv);
			if (mv.category === 'Status' || mv.selfdestruct || ONE_TIME_MOVES[mv.id]) return;
			var r = calc(ctx, me, opp, mv, {firstTurn: false, skipShield: opt.skipShield});
			var v = r.exp;
			if (mv.flags.recharge && !(ctx.field.gen === 1 && r.min >= opp.hp)) v *= 0.5; // Gen 1 skips the recharge on a KO
			if (mv.flags.charge && !(ctx.field.sun && (mv.id === 'solarbeam' || mv.id === 'solarblade')) && me.item !== 'powerherb') v *= 0.5;
			if (mv.id === 'suckerpunch' || mv.id === 'thunderclap') v *= opt.sucker === undefined ? 0.8 : opt.sucker;
			if (mv.id === 'futuresight' || mv.id === 'doomdesire') v *= 0.5;
			if (TYPE_BURN_MOVES[mv.id] && me.tera !== TYPE_BURN_MOVES[mv.id]) v *= 0.4; // works once
			// The matchup projection applies a move's own stat drops turn by turn, so it asks for
			// the undiscounted number.
			if (!opt.raw && mv.self && mv.self.boosts && (mv.self.boosts.spa < 0 || mv.self.boosts.atk < 0)) {
				var st = r.category === 'Special' ? mv.self.boosts.spa : mv.self.boosts.atk;
				if (st < 0) v *= st <= -2 ? 0.6 : 0.8;
			}
			if (v > out.exp) { out.exp = v; out.r = r; out.mv = mv; out.priority = r.priority; }
			if (r.priority > 0 && v > 0 && (!out.prio || v > out.prio.exp)) out.prio = {exp: v, r: r, mv: mv};
		});
		return out;
	}

	function withHp(c, hp) { return clone(c, {hp: clamp(hp, 0, c.maxhp)}); }

	var KO_BONUS = 0.15;
	var DUEL_TURNS = 14;

	// Fraction of max HP each recovery move restores.
	var HEAL_FRAC = {recover: 0.5, roost: 0.5, slackoff: 0.5, softboiled: 0.5, milkdrink: 0.5, healorder: 0.5, moonlight: 0.5, synthesis: 0.5, morningsun: 0.5, shoreup: 0.5, strengthsap: 0.4, rest: 0.6, wish: 0.4, lifedew: 0.25, junglehealing: 0.25, lunarblessing: 0.25};
	var SETUP_COST = {bellydrum: 0.5, filletaway: 0.5, clangoroussoul: 1 / 3};

	// How sure it is that the Pokemon has this move: certain for my own and for revealed moves,
	// otherwise the share of its possible sets that carry it.
	function hasMoveW(c, id) {
		if ((c.moves || []).indexOf(id) >= 0) return 1;
		if (c.mine) return 0;
		return c.poolW && c.poolW[id] ? Math.min(1, c.poolW[id]) : 0;
	}
	function statusBarred(c) {
		var v = c.volatiles || {};
		return !!v.taunt || c.item === 'assaultvest' || (c.itemKnown && /^choice(band|specs|scarf)$/.test(c.item || '') && !!c.lastMove);
	}

	// The recovery a Pokemon can fall back on: fraction of max HP per use, and how sure it is.
	function healOf(ctx, c) {
		if (statusBarred(c) || (c.volatiles || {}).healblock) return null;
		var f = ctx.field, best = null;
		for (var id in HEAL_FRAC) {
			var w = hasMoveW(c, id);
			if (w <= 0) continue;
			var frac = HEAL_FRAC[id];
			if (id === 'moonlight' || id === 'synthesis' || id === 'morningsun') frac = f.sun ? 2 / 3 : f.weather ? 0.25 : 0.5;
			if (id === 'shoreup' && f.sand) frac = 2 / 3;
			var nap = 0;
			if (id === 'rest' && cfg.restNap) {
				frac = 1;
				var chesto = c.itemKnown || c.mine ? c.item === 'chestoberry' : ((c.itemW || {}).chestoberry || 0) >= 0.5;
				nap = chesto || hasMoveW(c, 'sleeptalk') >= 0.5 ? 0 : 2;
			}
			if (!best || w * frac > best.w * best.frac) best = {id: id, frac: frac, w: w, nap: nap, rate: frac / (1 + nap)};
		}
		return best;
	}

	// The setup move a Pokemon would reach for, with the boosts it gives.
	function setupOf(ctx, c) {
		if (statusBarred(c)) return null;
		var ids = (c.moves || []).slice();
		if (!c.mine && c.poolW) for (var k in c.poolW) if (ids.indexOf(k) < 0) ids.push(k);
		var best = null;
		ids.forEach(function (id) {
			var mv = getMove(ctx.battle, id);
			if (mv.category !== 'Status') return;
			var b = id === 'bellydrum' ? {atk: 6} : id === 'curse' && c.types.indexOf('Ghost') >= 0 ? null : mv.boosts;
			if (!b || (mv.target !== 'self' && id !== 'bellydrum' && id !== 'curse')) return;
			var gain = 0, room = false;
			for (var s in b) {
				if (b[s] <= 0) continue;
				gain += b[s] * (s === 'atk' || s === 'spa' || s === 'spe' ? 1 : 0.5);
				if ((c.boosts[s] || 0) < 6) room = true;
			}
			var w = hasMoveW(c, id);
			if (gain <= 0 || !room || w <= 0) return;
			if (!best || w * gain > best.w * best.gain) best = {id: id, boosts: b, gain: gain, w: w, cost: SETUP_COST[id] || 0};
		});
		return best;
	}

	// The stat changes a move makes to its own user every time it lands.
	function selfBoostOf(mv) {
		if (!mv) return null;
		if (mv.self && mv.self.boosts) return mv.self.boosts;
		if (mv.selfBoost && mv.selfBoost.boosts) return mv.selfBoost.boosts;
		var sec = mv.secondary;
		if (sec && (sec.chance || 100) >= 100 && sec.self && sec.self.boosts) return sec.self.boosts;
		return null;
	}
	function withBoosts(c, b, herb) {
		var nb = Object.assign({}, c.boosts), flip = hasAb(c, 'contrary') ? -1 : 1, changed = false;
		for (var k in b) {
			var d = b[k] * flip;
			if (d < 0 && herb) continue; // White Herb undoes the drops once
			var v = clamp((nb[k] || 0) + d, -6, 6);
			if (v !== (nb[k] || 0)) changed = true;
			nb[k] = v;
		}
		return changed ? clone(c, {boosts: nb}) : c;
	}
	function boostKey(c) {
		var b = c.boosts || {};
		return (b.atk || 0) + ',' + (b.def || 0) + ',' + (b.spa || 0) + ',' + (b.spd || 0) + ',' + (b.spe || 0);
	}

	// Something that soaks up the first hit: a Substitute, Mimikyu's Disguise, Eiscue's Ice Face.
	function shieldOf(att, def, r, mv) {
		if (!r || !mv) return null;
		var dv = def.volatiles || {};
		if (dv.substitute && !mv.flags.sound && !mv.flags.bypasssub && !hasAb(att, 'infiltrator')) return {hp: def.maxhp / 4, chip: 0};
		if (r.moldBreaker || def.shieldDown) return null;
		var sid = def.species.id || '';
		if ((sid === 'mimikyu' || sid === 'mimikyutotem') && abChance(def, ['disguise']) > 0) return {hp: Infinity, chip: def.maxhp / 8};
		if (sid === 'eiscue' && abChance(def, ['iceface']) > 0 && r.category === 'Physical') return {hp: Infinity, chip: 0};
		return null;
	}

	// Does this Pokemon of mine still have a Disguise or an Ice Face up?
	function ownShield(c) {
		if (!c || c.shieldDown) return false;
		var sid = c.species.id || '';
		if ((sid === 'mimikyu' || sid === 'mimikyutotem') && abChance(c, ['disguise']) > 0 && !(c.volatiles || {}).disguisebroken) return true;
		return sid === 'eiscue' && abChance(c, ['iceface']) > 0;
	}

	// Chance the Pokemon acts on turn `t` of a projection. Sleep wears off after a couple of
	// turns; paralysis and the rest do not.
	function actAt(ctx, c, t) {
		var p = actChance(ctx, c);
		if (c.status === 'slp' && t > 2 && ctx.field.gen > 1) p = p / 0.33;
		return clamp(p, 0, 1);
	}

	// Heal now? Only when the hit after this one would finish me, healing gains on what comes in,
	// and I cannot simply take the KO instead.
	function wantsHeal(hp, max, incoming, heal, selfFirst, koNow) {
		if (koNow || heal <= incoming * 1.1 || hp >= max - 1) return false;
		return selfFirst ? hp <= incoming : (hp > incoming && hp <= 2 * incoming);
	}

	/**
	 * Plays the one-on-one forward turn by turn on expected damage. Both sides attack with their
	 * best move; either side heals when that keeps it alive; the opponent may open with `plan.k`
	 * turns of setup. Stat changes from the moves themselves (Close Combat, Draco Meteor, Aqua
	 * Step) are applied as they happen. `tieMeFirst` settles the turns where the order is a toss-up.
	 */
	function duel(ctx, me0, opp0, opt, plan, tieMeFirst, memo) {
		var meS = me0, oppS = opp0;
		var hpM = me0.hp, hpO = opp0.hp;
		var startM = hpM, startO = hpO;
		var myFrac = startM / me0.maxhp, oppFrac = startO / opp0.maxhp;
		var mult = opt.oppDmgMult === undefined ? 1 : opt.oppDmgMult;
		var get = function () {
			var key = boostKey(meS) + '|' + boostKey(oppS);
			if (memo[key]) return memo[key];
			var meX = withHp(meS, startM), oppX = withHp(oppS, startO);
			var atk = bestAttack(ctx, meX, oppX, {moves: opt.myMoves, ignoreLock: opt.ignoreLock, raw: true, skipShield: true, sucker: opt.sucker});
			var thr = threat(ctx, oppX, meX, {skipShield: true});
			var aM = actChance(ctx, meX), aO = actChance(ctx, oppX);
			var oPrio = null;
			thr.items.forEach(function (it) {
				if (it.r.priority > 0 && it.r.exp * it.w > 0 && (!oPrio || it.r.exp * it.w > oPrio.exp)) oPrio = {exp: it.r.exp * it.w / (aO || 1), priority: it.r.priority};
			});
			var n = {atk: atk, thr: thr, dM: atk.exp, dO: aO > 0 ? thr.exp / aO : 0, oPrio: oPrio, aM: aM, aO: aO};
			// What each side's own move costs or returns: recoil, Life Orb, drain.
			var amv = atk.mv, omv = thr.best ? thr.best.mv : null;
			n.selfM = 0; n.drainM = 0; n.selfO = 0; n.drainO = 0;
			if (amv) {
				if (amv.recoil && !hasAb(meX, 'rockhead', 'magicguard')) n.selfM += atk.exp * amv.recoil[0] / amv.recoil[1];
				if (meX.item === 'lifeorb' && !hasAb(meX, 'magicguard', 'sheerforce')) n.selfM += meX.maxhp / 10;
				if (amv.drain) n.drainM = atk.exp * amv.drain[0] / amv.drain[1];
			}
			if (omv) {
				if (omv.recoil && !hasAb(oppX, 'rockhead', 'magicguard')) n.selfO += n.dO * omv.recoil[0] / omv.recoil[1];
				var lo = oppX.itemKnown ? (oppX.item === 'lifeorb' ? 1 : 0) : ((oppX.itemW || {}).lifeorb || 0);
				if (!hasAb(oppX, 'magicguard', 'sheerforce')) n.selfO += lo * oppX.maxhp / 10;
				if (omv.drain) n.drainO = n.dO * omv.drain[0] / omv.drain[1];
			}
			n.boostM = selfBoostOf(amv);
			n.boostO = selfBoostOf(omv);
			return (memo[key] = n);
		};
		var first = get();
		var out = {atk: first.atk, thr: first.thr, fFrac: null};
		out.dMe = Math.max(0.01, first.dM * first.aM + residual(ctx, opp0) + (opt.myBonus || 0));
		out.dOpp = Math.max(0.01, first.thr.exp * mult + residual(ctx, me0) - (opt.myHeal || 0));
		out.f = firstProb(ctx, me0, opp0, first.atk.priority, first.thr.best ? first.thr.best.r.priority : 0);

		var stalled = false;
		var finish = function (win, turns) {
			out.win = win;
			out.nMe = out.nOpp = turns;
			if (win === 1) out.score = oppFrac + KO_BONUS - (startM - clamp(hpM, 0, me0.maxhp)) / me0.maxhp - (hpM <= 0 ? KO_BONUS : 0);
			else if (win === 0) out.score = (startO - clamp(hpO, 0, opp0.maxhp)) / opp0.maxhp - myFrac - KO_BONUS;
			else {
				// Nobody goes down in sight. Damage on a Pokemon that has been healing it off is
				// worth nothing; what counts is what the standoff costs me.
				var took = (startO - clamp(hpO, 0, opp0.maxhp)) / opp0.maxhp;
				if (stalled) took = Math.min(took, 0);
				out.win = stalled ? 0.3 : 0.5;
				out.score = 0.6 * (took - (startM - clamp(hpM, 0, me0.maxhp)) / me0.maxhp) - (stalled ? 0.1 : 0);
			}
			return out;
		};

		// Hits one side gets for free before the exchange starts (sleep, recharge and the like).
		if (opt.oppFreeHits) hpM -= opt.oppFreeHits * out.dOpp;
		if (opt.myFreeHits) hpO -= opt.myFreeHits * out.dMe;
		if (hpM <= 0) { hpO = Math.max(hpO, 0); return finish(0, 0); }
		if (hpO <= 0) return finish(1, 0);

		var healM = healOf(ctx, me0);
		var healO = plan && plan.heal ? healOf(ctx, opp0) : null;
		var oppHealed = false;
		var setup = plan && plan.setup ? plan.setup : null, kLeft = setup ? plan.k : 0;
		var herbO = opp0.item === 'whiteherb';
		var shield = shieldOf(me0, opp0, first.atk.r, first.atk.mv);
		// My own Disguise or Ice Face soaks their first hit the same way, and only the first.
		var shieldM = first.thr.best && ownShield(me0) ? shieldOf(opp0, me0, first.thr.best.r, first.thr.best.mv) : null;
		if (shieldM && shieldM.hp !== Infinity) shieldM = null;
		var sturdyM = hpM >= me0.maxhp ? Math.max(abChance(me0, ['sturdy']), me0.item === 'focussash' ? 1 : 0) : 0;
		var sturdyO = hpO >= opp0.maxhp ? Math.max(abChance(opp0, ['sturdy']), opp0.itemKnown ? (opp0.item === 'focussash' ? 1 : 0) : ((opp0.itemW || {}).focussash || 0)) : 0;
		var bumpsM = 0, bumpsO = 0;
		var napM = 0, napO = 0; // turns still asleep after a Rest
		var tie = function (p) {
			if (p <= 0) return false;
			if (p >= 1) return true;
			if (out.fFrac === null) out.fFrac = p;
			return tieMeFirst;
		};

		for (var t = 1; t <= DUEL_TURNS; t++) {
			var n = get();
			var dM = napM > 0 ? 0 : n.dM * actAt(ctx, meS, t);
			var dO = napO > 0 ? 0 : n.dO * actAt(ctx, oppS, t) * mult;
			var rM = residual(ctx, meS.status === 'tox' ? clone(meS, {toxicTurns: Math.max(-1, (meS.toxicTurns || 0) + t - 2)}) : meS) - (opt.myHeal || 0);
			var rO = residual(ctx, oppS.status === 'tox' ? clone(oppS, {toxicTurns: Math.max(-1, (oppS.toxicTurns || 0) + t - 2)}) : oppS) + (opt.myBonus || 0);
			var pMain = n.atk.priority, pOmain = n.thr.best ? n.thr.best.r.priority : 0;

			// ---- what each side does this turn
			var oppAct = napO > 0 ? 'nap' : 'atk', myAct = napM > 0 ? 'nap' : 'atk';
			var myKoNow = myAct === 'atk' && !shield && dM >= hpO && !(sturdyO >= 0.5);
			var oppKoNow = oppAct === 'atk' && !shieldM && dO >= hpM && !(sturdyM >= 0.5);
			// What a heal is worth: a Rest gives back the turns spent asleep after it.
			var inO = dM + Math.max(0, rO), inM = dO + Math.max(0, rM);
			if (oppAct === 'atk') {
				if (kLeft > 0 && hpO > opp0.maxhp * setup.cost + 1) oppAct = 'boost';
				else if (healO && wantsHeal(hpO, opp0.maxhp, inO, healO.frac * opp0.maxhp - (healO.nap || 0) * inO, !tie(firstProb(ctx, meS, oppS, pMain, 0)), oppKoNow)) oppAct = 'heal';
			}
			if (myAct === 'atk' && healM && wantsHeal(hpM, me0.maxhp, inM, healM.frac * me0.maxhp - (healM.nap || 0) * inM, tie(firstProb(ctx, meS, oppS, 0, pOmain)), myKoNow)) myAct = 'heal';

			var myHit = dM, myPrio = pMain, oppHit = dO, oppPrio = pOmain;
			if (myAct === 'atk' && n.atk.prio && n.atk.prio.mv !== n.atk.mv && !shield) {
				// A priority move when it finishes the job, or when it is the only hit I will get in.
				var pd = n.atk.prio.exp * actAt(ctx, meS, t);
				var outsped = !tie(firstProb(ctx, meS, oppS, pMain, oppAct === 'atk' ? pOmain : 0));
				if (pd >= hpO || (outsped && oppAct === 'atk' && oppKoNow)) { myHit = pd; myPrio = n.atk.prio.r.priority; }
			}
			if (oppAct === 'atk' && !shieldM && n.oPrio && n.oPrio.exp * actAt(ctx, oppS, t) * mult >= hpM && n.oPrio.priority > pOmain) {
				oppHit = n.oPrio.exp * actAt(ctx, oppS, t) * mult;
				oppPrio = n.oPrio.priority;
			}
			var meFirst = tie(firstProb(ctx, meS, oppS, myAct === 'atk' ? myPrio : 0, oppAct === 'atk' ? oppPrio : 0));

			// ---- play it out in order
			for (var step = 0; step < 2; step++) {
				var mine = (step === 0) === meFirst;
				if (mine) {
					if (myAct === 'nap') {
						// asleep after a Rest
					} else if (myAct === 'heal') {
						hpM = Math.min(me0.maxhp, hpM + healM.frac * me0.maxhp);
						if (healM.nap) napM = healM.nap + 1;
					} else {
						var hit = myHit;
						if (shield) {
							if (hit >= shield.hp || shield.hp === Infinity) { hit = shield.chip; shield = null; } else { shield.hp -= hit; hit = 0; }
						} else if (sturdyO >= 0.5 && hit >= hpO && !(n.atk.r && n.atk.r.hits > 1)) {
							hit = hpO - 1;
						}
						if (hit > 0 || myHit > 0) sturdyO = 0;
						hpO -= hit;
						hpM = Math.min(me0.maxhp, hpM + (myHit > 0 ? n.drainM : 0) - (myHit > 0 ? n.selfM : 0));
						if (n.boostM && myHit > 0 && bumpsM < 3) { bumpsM++; meS = withBoosts(meS, n.boostM, false); }
					}
					if (hpO <= 0) return finish(1, t);
					if (hpM <= 0) return finish(0, t);
				} else {
					if (oppAct === 'nap') {
						// asleep after a Rest
					} else if (oppAct === 'boost') {
						kLeft--;
						hpO -= opp0.maxhp * setup.cost;
						oppS = withBoosts(oppS, setup.boosts, herbO);
						herbO = false;
					} else if (oppAct === 'heal') {
						oppHealed = true;
						hpO = Math.min(opp0.maxhp, hpO + healO.frac * opp0.maxhp);
						if (healO.nap) napO = healO.nap + 1;
					} else {
						var oh = oppHit;
						if (shieldM && oh > 0) { oh = shieldM.chip; shieldM = null; }
						if (sturdyM >= 0.5 && oh >= hpM && !(n.thr.best && n.thr.best.r.hits > 1)) oh = hpM - 1;
						if (oh > 0) sturdyM = 0;
						hpM -= oh;
						hpO = Math.min(opp0.maxhp, hpO + (oppHit > 0 ? n.drainO : 0) - (oppHit > 0 ? n.selfO : 0));
						if (n.boostO && oppHit > 0 && bumpsO < 3) { bumpsO++; oppS = withBoosts(oppS, n.boostO, false); }
					}
					if (hpM <= 0) return finish(hpO <= 0 ? 1 : 0, t);
					if (hpO <= 0) return finish(1, t);
				}
			}
			// ---- end of turn
			if (napM > 0) napM--;
			if (napO > 0) napO--;
			hpM = Math.min(me0.maxhp, hpM - rM);
			hpO = Math.min(opp0.maxhp, hpO - rO);
			if (hpO <= 0) return finish(1, t);
			if (hpM <= 0) return finish(0, t);
		}
		stalled = oppHealed;
		return finish(0.5, DUEL_TURNS);
	}

	function mixRace(a, b, wb) {
		return clone(a, {score: (1 - wb) * a.score + wb * b.score, win: (1 - wb) * a.win + wb * b.win});
	}

	/**
	 * Projects the one-on-one from here and returns the expected material swing: the HP fraction
	 * the opponent loses minus the HP fraction I lose, with a bonus for whoever takes the KO.
	 * Positive means the exchange favours me. The opponent gets to pick the line that suits it
	 * best out of attacking, healing when it needs to, and setting up first, each weighted by how
	 * sure it is that the move is in its set.
	 */
	function race(ctx, me, opp, hpMe, hpOpp, opt) {
		opt = opt || {};
		var meC = withHp(me, hpMe), oppC = withHp(opp, hpOpp);
		var memo = {};
		var run = function (plan) {
			var a = duel(ctx, meC, oppC, opt, plan, true, memo);
			if (a.fFrac === null) return a;
			var b = duel(ctx, meC, oppC, opt, plan, false, memo);
			return mixRace(b, a, a.fFrac);
		};
		var base = run(null);
		if (!cfg.deepRace || opt.shallow) return base;
		var heal = healOf(ctx, oppC), su = setupOf(ctx, oppC);
		var pH = heal ? heal.w * cfg.healTrust : 0, pS = su ? su.w * cfg.setupTrust : 0;
		if (pH < 0.08) pH = 0;
		if (pS < 0.08) pS = 0;
		if (!pH && !pS) return base;
		var worst = function (list) { return list.reduce(function (x, y) { return y.score < x.score ? y : x; }); };
		var H = pH ? worst([base, run({heal: true})]) : base;
		var S = base, SH = H;
		if (pS) {
			S = worst([base, run({setup: su, k: 1}), run({setup: su, k: 2})]);
			if (pH) SH = worst([H, S, run({setup: su, k: 1, heal: true}), run({setup: su, k: 2, heal: true})]);
		}
		var w0 = (1 - pH) * (1 - pS), wH = pH * (1 - pS), wS = (1 - pH) * pS, wB = pH * pS;
		return clone(base, {
			plain: base.score,
			score: w0 * base.score + wH * H.score + wS * S.score + wB * SH.score,
			win: w0 * base.win + wH * H.win + wS * S.win + wB * SH.win,
			heals: pH, sets: pS
		});
	}

	// ------------------------------------------------------------------ option evaluation

	function oppKoChances(S, myPrio) {
		var before = 0, any = 0;
		S.thr.items.forEach(function (it) {
			var pko = koProb(it.r, S.me.hp, S.me) * it.w;
			var fMe = firstProb(S.ctx, S.me, S.opp, myPrio, it.r.priority);
			before = Math.max(before, pko * (1 - fMe));
			any = Math.max(any, pko);
		});
		return {before: before, any: any};
	}

	// Value of handing over to the best teammate after a pivot move.
	// What bringing `X0` in is worth when the opponent's move this turn lands on it. `hits` lists
	// what that move might be, each with its chance. A hit that kills is counted as a death with
	// that chance, instead of being blended into an average the Pokemon appears to survive.
	function entryValue(S, X0, hpOpp, hits, softer) {
		var ctx = S.ctx;
		var en = entryOf(ctx, X0, S.opp);
		var X = en.X, opp = en.opp;
		var hp0 = X.hp - en.dmg;
		var dead = -X.hp / X.maxhp - KO_BONUS;
		if (hp0 <= 0) return dead;
		var cont = function (hp, Xc) {
			if (hp <= 0) return dead;
			return -(X.hp - hp) / X.maxhp + race(ctx, Xc || X, opp, hp, Math.max(1, hpOpp), {ignoreLock: true}).score;
		};
		// A Disguise or an Ice Face that takes the hit on the way in is used up by it.
		var Xused = ownShield(X) ? clone(X, {shieldDown: true}) : null;
		var v = 0, left = 1;
		(hits || []).forEach(function (h) {
			var ph = Math.min(left, h.p);
			if (!(ph > 0) || !h.mv) return;
			var r = calc(ctx, opp, X, h.mv, {});
			var pk = koProb(r, hp0, withHp(X, hp0));
			var dmg = r.avg * r.acc * (1 - (r.immuneChance || 0));
			v += ph * (pk * dead + (1 - pk) * cont(hp0 - Math.min(dmg, hp0 - 1), Xused && h.mv.category !== 'Status' ? Xused : null));
			left -= ph;
		});
		// Otherwise it does something lighter: a weaker move, a status move, a switch of its own.
		if (left > 0.005) v += left * cont(hp0 - Math.min(softer || 0, hp0 - 1));
		return v;
	}

	// When my active Pokemon goes down, the next one comes in without taking a hit. How that
	// matchup goes is part of what staying in is worth: a switch is scored with the matchup that
	// follows it, so going down has to be scored with the one that follows it too. Otherwise the
	// bot throws a healthy Pokemon into a hit to save one that is already lost.
	function revengeOf(S) {
		if (S._rev !== undefined) return S._rev;
		var best = null;
		(S.bench || []).forEach(function (X0) {
			var v = entryValue(S, X0, S.opp.hp, [], 0);
			if (best === null || v > best) best = v;
		});
		S._rev = best === null ? 0 : cfg.revenge * clamp(best, -0.6, 0.9);
		return S._rev;
	}

	function pivotValue(S, hpOppAfter, oppHitsIncoming) {
		var best = -2;
		// Moving first, the pivot hands the incoming Pokemon the move they picked for this one.
		var hits = oppHitsIncoming && S.thr.best ? [{mv: S.thr.best.mv, p: S.thr.best.w}] : [];
		S.bench.forEach(function (X0) {
			var v = entryValue(S, X0, hpOppAfter, hits, 0);
			if (v > best) best = v;
		});
		return best;
	}

	function secondaryBonus(S, mv, r) {
		var me = S.me, opp = S.opp, b = 0;
		if (hasAb(me, 'sheerforce')) return 0;
		var secs = mv.secondaries || (mv.secondary ? [mv.secondary] : []);
		var grace = hasAb(me, 'serenegrace') ? 2 : 1;
		secs.forEach(function (s) {
			if (!s) return;
			var ch = Math.min(1, (s.chance || 100) / 100 * grace);
			if (s.status && !opp.status && !(opp.volatiles || {}).substitute) {
				var physical = S.thr.best && S.thr.best.r.category === 'Physical';
				var val = s.status === 'brn' ? (physical ? 0.2 : 0.05) : s.status === 'par' ? (effSpeed(S.ctx, opp) > effSpeed(S.ctx, me) ? 0.2 : 0.08) : s.status === 'frz' || s.status === 'slp' ? 0.3 : 0.06;
				if (s.status === 'brn' && opp.types.indexOf('Fire') >= 0) val = 0;
				if (s.status === 'par' && opp.types.indexOf('Electric') >= 0) val = 0;
				if ((s.status === 'psn' || s.status === 'tox') && (opp.types.indexOf('Poison') >= 0 || opp.types.indexOf('Steel') >= 0)) val = 0;
				b += ch * val;
			}
			if (s.boosts) b += ch * 0.04;
			if (s.self && s.self.boosts) b += ch * 0.05;
			if (s.volatileStatus === 'flinch') b += ch * 0.3 * clamp(S.thr.exp / me.maxhp, 0, 1) * firstProb(S.ctx, me, opp, r.priority, S.thr.best ? S.thr.best.r.priority : 0);
		});
		if (mv.id === 'knockoff' && (opp.item || !opp.itemKnown)) b += 0.05;
		if (mv.id === 'saltcure') b += 0.08;
		if (mv.id === 'rapidspin' || mv.id === 'mortalspin') b += myHazardWeight(S) * 0.07 * Math.min(3, S.bench.length);
		return b;
	}

	function myHazardWeight(S) {
		var sc = S.ctx.field.mySide || {};
		return (sc.stealthrock ? 1.2 : 0) + (sc.spikes ? 0.8 : 0) + (sc.toxicspikes ? 0.6 : 0) + (sc.stickyweb ? 0.7 : 0);
	}

	function evalAttack(S, a) {
		var ctx = S.ctx, me = S.me, opp = S.opp, mv = a.mv;
		var fBestGuess = S.thr.best ? firstProb(ctx, me, opp, movePriority(ctx, me, mv), S.thr.best.r.priority) : 1;
		var r = calc(ctx, me, opp, mv, {firstTurn: S.firstTurn, attFirst: fBestGuess >= 0.5, fallen: S.fallen});
		var act = actChance(ctx, me);
		var rel = mv.id === 'suckerpunch' || mv.id === 'thunderclap' ? S.sucker : mv.id === 'futuresight' || mv.id === 'doomdesire' ? 0.5 : 1;
		var charge = mv.flags.charge && !(ctx.field.sun && (mv.id === 'solarbeam' || mv.id === 'solarblade')) && !(mv.id === 'electroshot' && ctx.field.rain) && me.item !== 'powerherb' && !mv.isMaxed;
		if (charge) rel *= 0.5;
		var sub = (opp.volatiles || {}).substitute && !mv.flags.sound && !hasAb(me, 'infiltrator') && !mv.flags.bypasssub;
		var pk = sub || charge ? 0 : koProb(r, opp.hp, opp) * rel;
		var ko = oppKoChances(S, r.priority);
		// A faster Pokemon with a recovery move heals out of range before the hit lands. It does
		// that when it cannot simply take the KO first, or when a priority move would punish the
		// attack; and a turn spent healing is a turn it does not hit me.
		var healFirst = 0, healedTo = opp.hp;
		if (pk > 0 && S.oppHeal && !sub) {
			var fHeal = firstProb(ctx, me, opp, r.priority, 0);
			healedTo = Math.min(opp.maxhp, opp.hp + S.oppHeal.frac * opp.maxhp);
			var pkHealed = koProb(r, healedTo, clone(opp, {hp: healedTo})) * rel;
			if (fHeal < 1 && pkHealed < pk) {
				healFirst = (S.oppHeal.w >= 0.99 ? 1 : 0.5 * S.oppHeal.w) * cfg.healTrust * (1 - fHeal) * (opp.hp <= opp.maxhp * 0.75 ? 1 : 0.3);
				if ((S.prioKo || 0) < 0.3) healFirst *= 1 - ko.before;
				pk = pk * (1 - healFirst) + pkHealed * healFirst;
				ko = {before: ko.before * (1 - healFirst), any: ko.any * (1 - healFirst)};
			}
		}
		var noRetal = mv.id === 'fakeout' && !r.immune && S.firstTurn && !sub && !hasAb(opp, 'innerfocus');
		if (noRetal) ko = {before: 0, any: 0};
		var pivot = !!mv.selfSwitch && S.bench.length > 0 && !r.immune;
		var locks = !!(mv.self && mv.self.volatileStatus === 'lockedmove') && !mv.isMaxed;
		var fBest = S.thr.best ? firstProb(ctx, me, opp, r.priority, S.thr.best.r.priority) : 1;
		if (pivot) ko.any = ko.before;

		var pAct = (1 - ko.before) * act;
		var pKO = pAct * pk;
		var pDead = clamp(ko.before + (ko.any - ko.before) * (1 - pk * act), 0, 1);
		var pCont = Math.max(0, 1 - pKO - pDead);

		var dealt = sub ? 0 : Math.min(opp.hp, r.exp * rel) * act;
		// Against a Substitute only the hits after the one that breaks it get through.
		var subBroken = sub && r.exp * rel >= opp.maxhp / 4 / Math.max(1, r.hits);
		if (sub && subBroken && r.hits > 1) dealt = Math.min(opp.hp, r.exp * rel * (r.hits - 1) / r.hits) * act;
		var pOppActsOnMe = (noRetal ? 0 : pivot ? 1 - fBest : 1 - fBest * pk * act) * (1 - healFirst);
		var taken = S.thr.exp * pOppActsOnMe;

		var selfDmg = 0, heal = 0;
		if (mv.recoil && !hasAb(me, 'rockhead', 'magicguard')) selfDmg += dealt * mv.recoil[0] / mv.recoil[1];
		if (mv.hasCrashDamage) selfDmg += (1 - r.acc) * me.maxhp / 2;
		if (mv.mindBlownRecoil || mv.id === 'steelbeam' || mv.id === 'chloroblast') selfDmg += me.maxhp / 2;
		if (mv.id === 'struggle') selfDmg += me.maxhp / 4;
		if (me.item === 'lifeorb' && dealt > 0 && !hasAb(me, 'magicguard', 'sheerforce')) selfDmg += me.maxhp / 10;
		if (mv.flags.contact && dealt > 0) {
			if (opp.item === 'rockyhelmet') selfDmg += me.maxhp / 6;
			selfDmg += abChance(opp, ['roughskin', 'ironbarbs']) * me.maxhp / 8;
		}
		if (mv.drain) heal += dealt * mv.drain[0] / mv.drain[1];

		var hpMeAfter = clamp(me.hp - taken - selfDmg + heal, 0, me.maxhp);
		var hpOppAfter = opp.hp + healFirst * (healedTo - opp.hp) - dealt;
		// Chip damage on something that heals more than I deal each turn does not last.
		var sticks = 1;
		if (S.oppHeal && pk < 0.5 && r.exp * rel + Math.max(0, residual(ctx, opp)) < (S.oppHeal.rate || S.oppHeal.frac) * opp.maxhp * 0.9) {
			sticks = 1 - 0.8 * S.oppHeal.w * cfg.healTrust;
		}
		// Life Orb or recoil that takes my last HP: the move costs me the Pokemon as well. So does
		// poison, Salt Cure or Leech Seed when it is about to take what is left.
		var chip = Math.max(0, residual(ctx, me));
		hpMeAfter = clamp(hpMeAfter - chip, 0, me.maxhp);
		var suicide = selfDmg - heal + chip >= me.hp;
		var myFrac = me.hp / me.maxhp, oppFrac = opp.hp / opp.maxhp;
		var lostThisTurn = (me.hp - hpMeAfter) / me.maxhp;
		// I take the KO: they lose what they had left, I lose whatever landed before it.
		var winV = oppFrac + KO_BONUS + boostWorth(opp) - clamp(((1 - fBest) * S.thr.exp + selfDmg - heal) / me.maxhp, -1, myFrac) - (suicide ? KO_BONUS : 0);
		// I go down: still count any damage I got in first.
		var dealtIfDead = pDead > 0 ? (pDead - ko.before) / pDead * dealt : 0;
		var loseV = dealtIfDead / opp.maxhp - myFrac - KO_BONUS;

		var V;
		if (mv.selfdestruct || mv.id === 'finalgambit') {
			// My Pokemon is gone either way, so this is only worth what it takes with it.
			var boom = pk * (oppFrac + KO_BONUS) + (1 - pk) * Math.min(opp.hp, r.exp) / opp.maxhp;
			V = pAct * boom - myFrac - KO_BONUS;
		} else {
			var contV;
			if (suicide) {
				contV = dealt / opp.maxhp - myFrac - KO_BONUS;
			} else if (hpMeAfter <= 0) {
				contV = loseV;
			} else if (pivot) {
				contV = dealt * sticks / opp.maxhp - lostThisTurn + pivotValue(S, hpOppAfter, fBest >= 0.5) - 0.03;
				// Asleep or fully paralysed, the move never happens and I am still in.
				if (act < 1) contV = act * contV + (1 - act) * (-S.thr.exp / me.maxhp + race(ctx, me, opp, Math.max(1, me.hp - S.thr.exp), opp.hp, {sucker: S.sucker}).score);
			} else {
				var me2 = withHp(me, hpMeAfter);
				var sb = mv.isMaxed ? null : selfBoostOf(mv);
				if (sb && !r.immune) me2 = withHp(withBoosts(me, sb, false), hpMeAfter);
				// My Disguise or Ice Face took their hit this turn and is gone after it.
				if (ownShield(me) && pOppActsOnMe >= 0.5) me2 = clone(me2, {shieldDown: true});
				var extra = {sucker: S.sucker};
				if (mv.flags.recharge && !mv.isMaxed) extra.oppFreeHits = 1; // this branch is the one where it did not KO
				// A Choice item locks me into this move for as long as I stay in, and so does a
				// move like Outrage.
				if ((/^choice(band|specs|scarf)$/.test(me.item) || locks) && !mv.isMaxed) extra.myMoves = [mv.id];
				if (charge) extra.oppFreeHits = 1;
				// If it Rested out of range this turn, it is asleep for the two turns after.
				if (healFirst > 0 && S.oppHeal && S.oppHeal.nap) extra.myFreeHits = (extra.myFreeHits || 0) + healFirst * S.oppHeal.nap;
				var oppNext = subBroken ? clone(opp, {volatiles: Object.assign({}, opp.volatiles, {substitute: null})}) : opp;
				// The hit that breaks a Disguise or an Ice Face leaves it broken.
				if (!sub && !r.immune && shieldOf(me, opp, r, mv)) oppNext = clone(oppNext, {shieldDown: true});
				var after = race(ctx, me2, oppNext, hpMeAfter, Math.max(1, hpOppAfter), extra).score;
				// A move that lowers the target's stats changes how the rest of the fight goes.
				var drop = r.immune || r.exp <= 0 ? null : targetDrop(S, mv);
				if (drop) after = drop.ch * race(ctx, me2, drop.opp, hpMeAfter, Math.max(1, hpOppAfter), extra).score + (1 - drop.ch) * after;
				contV = dealt * sticks / opp.maxhp - lostThisTurn + after;
			}
			V = pKO * winV + pDead * loseV + pCont * contV;
			V += secondaryBonus(S, mv, r) * pAct;
			if (sub && r.exp > 0) V += 0.1;
		}
		if ((r.immune || r.exp <= 0) && !(!r.immune && !sub && shieldOf(me, opp, r, mv))) V -= 0.35;
		else V += 0.02 * clamp(r.exp / opp.maxhp, 0, 1);
		// A Choice item keeps me on this move after the KO as well. What that costs against the
		// rest of their team counts, and counts in full when there is nobody left to switch to.
		if (/^choice(band|specs|scarf)$/.test(me.item) && !mv.isMaxed && S.oppAlive > 1 && pKO > 0.3 && S.benchOpp && S.benchOpp.length) {
			var worse = 0;
			S.benchOpp.forEach(function (Y) {
				var held = Math.min(1, calc(ctx, me, Y, mv, {firstTurn: false}).exp / Math.max(1, Y.hp));
				var free = Math.min(1, bestAttack(ctx, me, Y, {ignoreLock: true, raw: true}).exp / Math.max(1, Y.hp));
				worse += Math.min(0, held - free);
			});
			V += pKO * clamp(worse / S.benchOpp.length, -0.5, 0) * (S.bench.length ? 0.4 : 1.2);
		}
		// Locked in for two or three turns: whatever they bring in next gets to deal with one
		// move, and something immune to it gets a free turn.
		if (locks && S.oppAlive > 1 && !(me.volatiles || {}).lockedmove) {
			var pen = cfg.lockPenalty * (1 + 0.25 * Math.max(0, S.oppAlive - 2)), least = 1;
			(S.benchOpp || []).forEach(function (Y) {
				var ry = calc(ctx, me, Y, mv, {firstTurn: false});
				least = Math.min(least, ry.immune ? 0 : ry.exp / Y.maxhp);
			});
			if (least <= 0.01) pen += 0.2; else if (least < 0.15) pen += 0.1;
			V -= pen;
		}
		return {V: V, pKO: pKO, pDead: pDead, dealt: dealt / opp.maxhp, r: r, suicide: suicide, healFirst: healFirst};
	}

	// Chance the opponent attacks this turn, which is what Sucker Punch needs to work.
	function suckerRel(S) {
		var opp = S.opp, me = S.me;
		var statusy = (opp.moves || []).filter(function (m) { return getMove(S.ctx.battle, m).category === 'Status'; }).length;
		var p = statusy ? 0.65 : opp.moves.length >= 3 ? 0.92 : 0.8;
		// Low on HP with a recovery move, or sitting on a setup move with nothing to fear.
		if (S.oppHeal && S.oppHeal.w >= 0.5 && opp.hp < opp.maxhp * 0.6) p = Math.min(p, 0.45);
		if (S.oppSetup && S.oppSetup.w >= 0.5 && S.thr.exp < me.hp * 0.5) p = Math.min(p, 0.55);
		var fails = (((S.st || {}).suckerFails || {})[toID((opp.ident || '').replace(/^p\d: /, '') || opp.name)]) || 0;
		if (fails) p *= Math.pow(0.5, fails);
		return clamp(p, 0.08, 0.95);
	}

	var PROTECT_MOVES = {protect: 1, detect: 1, kingsshield: 1, spikyshield: 1, banefulbunker: 1, obstruct: 1, silktrap: 1, burningbulwark: 1, maxguard: 1, endure: 1};
	var HAZARD_VALUE = {stealthrock: 0.09, spikes: 0.06, toxicspikes: 0.05, stickyweb: 0.06};
	var HAZARD_MAX = {stealthrock: 1, spikes: 3, toxicspikes: 2, stickyweb: 1};
	var WEATHER_MOVES = {sunnyday: 'sun', raindance: 'rain', sandstorm: 'sand', snowscape: 'snow', hail: 'hail'};

	// Chance a status move actually takes effect on the opponent.
	function statusLands(ctx, me, opp, mv) {
		var gen = ctx.field.gen;
		var p = mv.accuracy === true ? 1 : (mv.accuracy || 100) / 100;
		var ov = opp.volatiles || {};
		if (ov.substitute && !mv.flags.sound && !mv.flags.bypasssub) return 0;
		if (hasAb(me, 'prankster') && gen >= 7 && opp.types.indexOf('Dark') >= 0) return 0;
		if (mv.flags.powder && gen >= 6 && (opp.types.indexOf('Grass') >= 0 || opp.item === 'safetygoggles' || hasAb(opp, 'overcoat'))) return 0;
		if (mv.id === 'thunderwave') {
			var e = 1;
			opp.types.forEach(function (t) { e *= typeMult(ctx.battle, 'Electric', t); });
			if (e === 0) return 0;
		}
		p *= 1 - abChance(opp, ['goodasgold', 'magicbounce']);
		var st = mv.status || (mv.id === 'yawn' ? 'slp' : '');
		if (st) {
			if (opp.status) return 0;
			if ((sideOf(ctx, opp) || {}).safeguard) return 0;
			if (ctx.field.terrain === 'misty' && opp.grounded) return 0;
			if (st === 'par' && gen >= 6 && opp.types.indexOf('Electric') >= 0) return 0;
			if (st === 'brn' && opp.types.indexOf('Fire') >= 0) return 0;
			if ((st === 'psn' || st === 'tox') && (opp.types.indexOf('Poison') >= 0 || opp.types.indexOf('Steel') >= 0) && !hasAb(me, 'corrosion')) return 0;
			if (st === 'slp') {
				if (ctx.field.terrain === 'electric' && opp.grounded) return 0;
				var asleep = ((ctx.battle.farSide && ctx.battle.farSide.pokemon) || []).some(function (q) { return q.status === 'slp' && !q.fainted; });
				if (asleep) return 0; // Sleep Clause
				p *= 1 - abChance(opp, ['insomnia', 'vitalspirit', 'sweetveil', 'comatose']);
			}
			if (st === 'par') p *= 1 - abChance(opp, ['limber', 'comatose']);
			if (st === 'brn') p *= 1 - abChance(opp, ['waterveil', 'thermalexchange', 'waterbubble', 'comatose', 'guts']);
			if (st === 'psn' || st === 'tox') p *= 1 - abChance(opp, ['immunity', 'poisonheal', 'comatose', 'guts', 'magicguard', 'pastelveil']);
			p *= 1 - abChance(opp, ['purifyingsalt']);
			if (ctx.field.sun) p *= 1 - abChance(opp, ['leafguard']);
		}
		return clamp(p, 0, 1);
	}

	function evalStatus(S, a) {
		var ctx = S.ctx, me = S.me, opp = S.opp, mv = a.mv, id = mv.id, f = ctx.field;
		var act = actChance(ctx, me);
		var prio = movePriority(ctx, me, mv);
		var fBest = S.thr.best ? firstProb(ctx, me, opp, prio, S.thr.best.r.priority) : 1;
		var ko = oppKoChances(S, prio);
		var T = S.thr.exp;
		var loseV = -me.hp / me.maxhp - KO_BONUS;
		var FAIL = {V: -3, fail: true};
		// Material change this turn, plus how the matchup plays out from the resulting state.
		var cont = function (me2, opp2, hpMe2, hpOpp2, extra, ctx2) {
			if (hpMe2 <= 0) return loseV;
			hpMe2 = Math.min(hpMe2, me2.maxhp);
			// My Disguise or Ice Face is spent by the hit of this turn; the turns after start without it.
			if (ownShield(me2)) me2 = clone(me2, {shieldDown: true});
			return (opp.hp - hpOpp2) / opp.maxhp - (me.hp - hpMe2) / me.maxhp + race(ctx2 || ctx, me2, opp2, hpMe2, Math.max(1, hpOpp2), extra).score;
		};
		var wrap = function (v, pDead) {
			pDead = pDead === undefined ? ko.any : pDead;
			return pDead * loseV + (1 - pDead) * v;
		};
		var plain = function () { return cont(me, opp, me.hp - T, opp.hp); };
		var mix = function (v) { return {V: act * v + (1 - act) * wrap(plain())}; };
		var mv_boosts = mv.boosts;

		if (me.status === 'slp' && id !== 'sleeptalk') return {V: wrap(plain()) - 0.05};

		// --- Protect and friends
		if (PROTECT_MOVES[id] || mv.stallingMove) {
			if (PROTECT_MOVES[me.lastMove]) return FAIL;
			var wish = ctx.battle.nearSide && ctx.battle.nearSide.wisher ? me.maxhp / 2 : 0;
			var pv = cont(me, opp, Math.min(me.maxhp, me.hp - residual(ctx, me) + wish), opp.hp - residual(ctx, opp));
			// A turn of Protect leaves the matchup where it was. What it buys is the residual
			// damage and healing of one turn; what it costs is the turn, and a free boost for
			// anything that wants one.
			var gain = residual(ctx, opp) / opp.maxhp + (Math.min(me.maxhp - me.hp, wish) - residual(ctx, me)) / me.maxhp;
			var cost = 0.12 + (S.oppSetup ? 0.3 * S.oppSetup.w : 0) + (S.oppHeal && opp.hp < opp.maxhp ? 0.15 * S.oppHeal.w : 0);
			return {V: pv - 0.08 + (opp.dynamaxed ? 0.25 : 0), protect: gain - cost + (opp.dynamaxed ? 0.25 : 0)};
		}

		// --- Belly Drum and other self-boosting moves
		if (id === 'bellydrum') {
			if (me.hp * 2 <= me.maxhp || (me.boosts.atk || 0) >= 6) return FAIL;
			mv_boosts = {atk: 6};
		}
		if (id === 'growth' && f.sun) mv_boosts = {atk: 2, spa: 2};
		if (id === 'curse' && me.types.indexOf('Ghost') >= 0) return {V: wrap(plain()) - 0.3};
		if (mv_boosts && (mv.target === 'self' || id === 'bellydrum' || id === 'curse')) {
			var me2 = clone(me, {boosts: Object.assign({}, me.boosts)});
			var changed = false, top = 0;
			for (var k in mv_boosts) {
				var before = me2.boosts[k] || 0;
				var after = clamp(before + mv_boosts[k], -6, 6);
				if (after > before) { changed = true; top = Math.max(top, before); }
				me2.boosts[k] = after;
			}
			if (!changed) return FAIL;
			var cost = id === 'bellydrum' || id === 'filletaway' ? me.maxhp / 2 : id === 'clangoroussoul' ? me.maxhp / 3 : 0;
			if (cost && me.hp <= cost) return FAIL;
			// White Herb undoes the drops that come with the boost (Shell Smash), once.
			if (me.item === 'whiteherb') {
				for (var wk2 in me2.boosts) if (me2.boosts[wk2] < (me.boosts[wk2] || 0)) me2.boosts[wk2] = me.boosts[wk2] || 0;
			}
			var thrUp = fBest >= 0.5 ? threat(ctx, opp, me2) : S.thr;
			var tTurn = thrUp.exp;
			// Setting up is a bet on surviving the turn. If the move it probably has does much
			// more than the average says, plan for that hit.
			if (thrUp.best && thrUp.best.w >= 0.5 && thrUp.best.r.avg * thrUp.best.r.acc > tTurn) tTurn = 0.5 * tTurn + 0.5 * thrUp.best.r.avg * thrUp.best.r.acc;
			var hpUp = me.hp - cost - tTurn;
			var sv = cont(me2, opp, hpUp, opp.hp);
			var sweep = sweepGain(S, me, me2, hpUp);
			// The Pokemon in front of me can undo the boost (Haze, Roar, Encore, Unaware and so
			// on): then the turn buys nothing here and nothing later.
			var undo = abChance(opp, ['unaware']);
			for (var ba in BOOST_ANSWERS) undo = Math.max(undo, hasMoveW(opp, ba));
			if (undo > 0) {
				sv = (1 - undo) * sv + undo * (cont(me, opp, hpUp, opp.hp) - 0.05);
				sweep *= 1 - undo;
			}
			// The hit I take while setting up lands on what is left after the move's own cost.
			var hpAtHit = me.hp - cost, pDeadUp = 0;
			thrUp.items.forEach(function (it) { pDeadUp = Math.max(pDeadUp, koProb(it.r, hpAtHit, clone(me2, {hp: hpAtHit})) * it.w); });
			if (T > me.maxhp * 0.4) sweep *= 0.35; // taking a big hit to set up rarely leaves enough to sweep with
			if (me.status === 'tox' || me.status === 'psn' || (me.volatiles || {}).perish3 || (me.volatiles || {}).leechseed) sweep *= 0.6;
			sv += sweep;
			if (!(cfg.sweepWeight > 0) && sv > 0) sv += 0.05 * Math.min(4, Math.max(0, S.oppAlive - 1));
			if (top >= 4) sv -= 0.25; else if (top >= 2 && T > me.maxhp * 0.2) sv -= 0.12;
			if (T > me.maxhp * 0.4) sv -= 0.3; // too dangerous to spend the turn boosting
			var out = mix(wrap(sv, Math.max(ko.any, pDeadUp)) - 0.04);
			out.setup = true;
			if (id === 'tidyup') out.V += myHazardWeight(S) * 0.07 * Math.min(3, S.bench.length);
			return out;
		}

		// --- recovery
		var heals = mv.heal || mv.flags.heal || id === 'rest' || id === 'wish' || id === 'painsplit' || id === 'strengthsap';
		if (heals && mv.target !== 'normal' || id === 'painsplit' || id === 'strengthsap') {
			if ((me.volatiles || {}).healblock) return FAIL;
			var amt = me.maxhp / 2;
			if (mv.heal) amt = me.maxhp * mv.heal[0] / mv.heal[1];
			if (id === 'moonlight' || id === 'synthesis' || id === 'morningsun') amt = me.maxhp * (f.sun ? 2 / 3 : f.weather ? 1 / 4 : 1 / 2);
			if (id === 'shoreup' && f.sand) amt = me.maxhp * 2 / 3;
			if (id === 'rest') amt = me.maxhp;
			if (id === 'painsplit') amt = (opp.hp - me.hp) / 2;
			if (id === 'strengthsap') amt = (opp.stats.atk || 1) * boostMult(opp.boosts.atk);
			// Strength Sap also takes a stage off the target's Attack, which is worth using at any
			// HP against something physical.
			var sapped = null;
			if (id === 'strengthsap' && (opp.boosts.atk || 0) > -6 && !(opp.volatiles || {}).substitute && abChance(opp, ['clearbody', 'whitesmoke', 'fullmetalbody', 'contrary', 'defiant']) < 0.5) {
				sapped = withBoosts(opp, {atk: -1}, false);
			}
			var sapWorth = sapped && S.thr.best && S.thr.best.r.category === 'Physical';
			if ((me.hp > me.maxhp * 0.7 && !sapWorth) || amt <= 0) return FAIL;
			var hp2, pDead;
			if (id === 'wish') {
				hp2 = me.hp - 2 * T + amt;
				pDead = ko.any;
			} else if (fBest >= 0.5) {
				var healed = Math.min(me.maxhp, me.hp + amt);
				hp2 = healed - T;
				pDead = 0;
				S.thr.items.forEach(function (it) { pDead = Math.max(pDead, koProb(it.r, healed, clone(me, {hp: healed})) * it.w); });
			} else {
				hp2 = Math.min(me.maxhp, me.hp - T + amt);
				pDead = ko.any;
			}
			var meH = id === 'rest' ? clone(me, {status: ''}) : me;
			var hv = cont(meH, sapped || opp, hp2, id === 'painsplit' ? opp.hp - amt : opp.hp, id === 'rest' ? {oppFreeHits: 2} : {});
			if (amt <= T * 1.05 && residual(ctx, opp) <= 0 && !sapWorth) hv -= 0.15; // healing less than the incoming hit goes nowhere
			// A faster Psychic Noise blocks the heal outright.
			if (fBest < 0.5 && (opp.moves || []).indexOf('psychicnoise') >= 0) hv -= 0.2;
			var hOut = mix(wrap(hv, pDead) - 0.03);
			hOut.heal = true;
			return hOut;
		}

		// --- status on the opponent
		if ((mv.status || id === 'yawn') && mv.target !== 'self') {
			var st = mv.status || 'slp';
			var s = statusLands(ctx, me, opp, mv) * (id === 'yawn' ? 0.7 : 1);
			if (s <= 0) return FAIL;
			var opp2 = clone(opp, {status: st, toxicTurns: 0});
			var extra = {};
			var hpMe2 = me.hp - T;
			var pD = ko.any;
			if (st === 'slp') {
				extra.myFreeHits = f.gen === 1 ? 3 : 1;
				if (fBest >= 0.5 && id !== 'yawn') { hpMe2 = me.hp - (1 - s) * T; pD = ko.any * (1 - s); }
			}
			var vS = cont(me, opp2, hpMe2, opp.hp, extra);
			// Poison keeps working after this matchup is over, and it is what beats a Pokemon that
			// means to heal or boost its way through.
			if ((st === 'tox' || st === 'psn') && (S.oppHeal || S.oppSetup)) vS += 0.18 * Math.max(S.oppHeal ? S.oppHeal.w : 0, S.oppSetup ? S.oppSetup.w : 0);
			var vF = plain();
			return mix(wrap(s * vS + (1 - s) * vF, pD) - 0.02);
		}

		// --- entry hazards
		if (mv.sideCondition && HAZARD_VALUE[mv.sideCondition]) {
			var kind = mv.sideCondition;
			var cur = f.oppSide[kind];
			var layers = cur ? (cur[1] || 1) : 0;
			var left = S.oppAlive - 1;
			if (layers >= HAZARD_MAX[kind] || left <= 0 || abChance(opp, ['magicbounce']) >= 1) return FAIL;
			var hz = HAZARD_VALUE[kind] * Math.min(4, left) * (layers ? 0.6 : 1);
			return mix(wrap(plain() + hz));
		}

		// --- hazard removal
		if (id === 'defog') {
			var hw = myHazardWeight(S);
			var os = f.oppSide || {};
			var screens = os.reflect || os.lightscreen || os.auroraveil ? 0.15 : 0;
			if (!hw && !screens) return FAIL;
			var mine = (os.stealthrock ? 0.09 : 0) + (os.spikes ? 0.06 : 0) + (os.toxicspikes ? 0.05 : 0) + (os.stickyweb ? 0.06 : 0);
			return mix(wrap(plain() + hw * 0.07 * Math.min(3, S.bench.length) + screens - mine * Math.min(4, S.oppAlive - 1)));
		}

		// --- screens
		if (id === 'reflect' || id === 'lightscreen' || id === 'auroraveil') {
			if (f.mySide[id]) return FAIL;
			if (id === 'auroraveil' && !(f.snow || f.hail)) return FAIL;
			var bcat = S.thr.best ? S.thr.best.r.category : '';
			var relevant = id === 'auroraveil' || (id === 'reflect' && bcat === 'Physical') || (id === 'lightscreen' && bcat === 'Special');
			var scv = relevant ? cont(me, opp, me.hp - (fBest >= 0.5 ? T / 2 : T), opp.hp, {oppDmgMult: 0.5}) : plain() - 0.15;
			return mix(wrap(scv + 0.03 * S.bench.length));
		}

		// --- removing the opponent's boosts
		if (id === 'haze' || id === 'roar' || id === 'whirlwind') {
			var bsum = 0;
			['atk', 'spa', 'spe', 'def', 'spd'].forEach(function (k2) { if ((opp.boosts[k2] || 0) > 0) bsum += opp.boosts[k2]; });
			if (bsum < 2) return {V: wrap(plain()) - 0.3};
			if (id !== 'haze' && S.oppAlive <= 1) return FAIL;
			var oppC = clone(opp, {boosts: {}});
			var blocked = abChance(opp, id === 'roar' ? ['goodasgold', 'soundproof', 'suctioncups'] : id === 'whirlwind' ? ['goodasgold', 'suctioncups'] : []);
			if (blocked >= 0.99 || (id !== 'haze' && (opp.volatiles || {}).ingrain)) return FAIL;
			var pv = cont(me, oppC, me.hp - T, opp.hp) + 0.05;
			// The boosts were going to cost more than this one matchup: count what they would
			// have done to the rest of my team.
			if (cfg.phazeTeam > 0 && S.bench.length) {
				var saved = 0;
				S.bench.forEach(function (X) {
					var withB = race(ctx, X, opp, X.hp, opp.hp, {ignoreLock: true}).score;
					var noB = race(ctx, X, oppC, X.hp, opp.hp, {ignoreLock: true}).score;
					saved += Math.max(0, noB - withB);
				});
				pv += cfg.phazeTeam * Math.min(saved, 2.5);
			}
			pv = (1 - blocked) * pv + blocked * plain();
			return mix(wrap(pv));
		}

		if (id === 'taunt') {
			var statusy = opp.moves.filter(function (m) { return getMove(ctx.battle, m).category === 'Status'; }).length;
			var passive = opp.moves.length && statusy * 2 >= opp.moves.length;
			return mix(wrap(plain() + (passive ? 0.2 : -0.25)));
		}

		if (id === 'encore') {
			if (!opp.lastMove || (opp.volatiles || {}).encore || opp.dynamaxed) return FAIL;
			var lm = getMove(ctx.battle, opp.lastMove);
			var lr = calc(ctx, opp, me, lm, {});
			var weak = lm.category === 'Status' || lr.exp < me.maxhp * 0.12;
			if (!weak) return {V: wrap(plain()) - 0.2};
			var oppE = clone(opp, {forcedMove: opp.lastMove});
			return mix(wrap(cont(me, oppE, me.hp - (fBest >= 0.5 ? lr.exp : T), opp.hp) + 0.05, fBest >= 0.5 ? 0 : ko.any));
		}

		if (id === 'leechseed') {
			if (opp.types.indexOf('Grass') >= 0 || (opp.volatiles || {}).leechseed || (opp.volatiles || {}).substitute) return FAIL;
			var ls = (mv.accuracy === true ? 1 : mv.accuracy / 100) * (1 - abChance(opp, ['magicbounce', 'goodasgold']));
			var oppL = clone(opp, {volatiles: Object.assign({}, opp.volatiles, {leechseed: ['leechseed']})});
			return mix(wrap(ls * cont(me, oppL, me.hp - T, opp.hp, {myHeal: opp.maxhp / 8}) + (1 - ls) * plain()));
		}

		if (id === 'substitute') {
			if ((me.volatiles || {}).substitute || me.hp * 4 <= me.maxhp) return FAIL;
			var safe = opp.status === 'slp' || opp.status === 'frz' || T < me.maxhp * 0.25;
			return mix(wrap(plain() + (safe ? 0.08 : -0.3)));
		}

		if (id === 'trick' || id === 'switcheroo') {
			if (!/^choice/.test(me.item) || (opp.itemKnown && !opp.item)) return FAIL;
			return mix(wrap(plain() + 0.18));
		}

		if (id === 'destinybond') {
			if (ko.any > 0.5 && fBest >= 0.5 && me.lastMove !== 'destinybond') return {V: ko.any * (opp.hp / opp.maxhp - me.hp / me.maxhp) + (1 - ko.any) * (wrap(plain()) - 0.2)};
			return {V: -1};
		}

		// --- lowering the opponent's stats (Charm, Screech, Parting Shot and the like)
		var oppLow = null;
		if (mv.boosts && mv.target !== 'self' && id !== 'curse' && id !== 'bellydrum') {
			var lb = Object.assign({}, opp.boosts), lowered = false;
			for (var lk in mv.boosts) {
				if (mv.boosts[lk] >= 0) continue;
				var lv = clamp((lb[lk] || 0) + mv.boosts[lk], -6, 6);
				if (lv !== (lb[lk] || 0)) lowered = true;
				lb[lk] = lv;
			}
			if (lowered) {
				var land = (mv.accuracy === true ? 1 : mv.accuracy / 100) * (1 - abChance(opp, ['magicbounce', 'goodasgold', 'clearbody', 'whitesmoke', 'fullmetalbody', 'mirrorarmor']));
				if ((opp.volatiles || {}).substitute && !mv.flags.sound) land = 0;
				if (opp.item === 'clearamulet') land = 0;
				if (land <= 0) return FAIL;
				// Lowering these feeds the very thing it is meant to stop.
				if (abChance(opp, ['contrary', 'defiant', 'competitive']) >= 0.5) return {V: wrap(plain()) - 0.4};
				oppLow = {opp: clone(opp, {boosts: lb}), land: land};
				if (!mv.selfSwitch) {
					var tLow = fBest >= 0.5 ? threat(ctx, oppLow.opp, me).exp : T;
					return mix(wrap(land * cont(me, oppLow.opp, me.hp - tLow, opp.hp) + (1 - land) * plain()) - 0.02);
				}
			}
		}

		// --- pivoting status moves
		if (mv.selfSwitch) {
			if (!S.bench.length) return FAIL;
			// Parting Shot hands over to a teammate against a weakened opponent.
			var Sp = oppLow ? clone(S, {opp: oppLow.opp, thr: threat(ctx, oppLow.opp, me)}) : S;
			var pvv = pivotValue(Sp, opp.hp, fBest >= 0.5);
			var hpLeft = fBest >= 0.5 ? me.hp : me.hp - T;
			if (id === 'shedtail' && me.hp * 2 <= me.maxhp) return FAIL;
			return {V: (hpLeft <= 0 ? loseV : ko.before * loseV + (1 - ko.before) * (pvv - (me.hp - hpLeft) / me.maxhp)) - 0.04};
		}

		if (id === 'trickroom') {
			var ctxT = clone(ctx, {field: clone(f, {trickRoom: !f.trickRoom})});
			return mix(wrap(cont(me, opp, me.hp - T, opp.hp, {}, ctxT)) - 0.03);
		}
		if (id === 'tailwind') {
			if (f.mySide.tailwind) return FAIL;
			var ctxW = clone(ctx, {field: clone(f, {mySide: Object.assign({}, f.mySide, {tailwind: ['Tailwind']})})});
			return mix(wrap(cont(me, opp, me.hp - T, opp.hp, {}, ctxW)) - 0.03);
		}

		if (WEATHER_MOVES[id]) {
			var wk = WEATHER_MOVES[id];
			if (f[wk]) return FAIL;
			var syn = (wk === 'rain' && hasAb(me, 'swiftswim', 'raindish', 'dryskin')) || (wk === 'sun' && hasAb(me, 'chlorophyll', 'solarpower')) || (wk === 'sand' && hasAb(me, 'sandrush', 'sandforce')) || (wk === 'snow' && hasAb(me, 'slushrush'));
			if (wk === 'rain' && me.moves.some(function (m) { return m === 'thunder' || m === 'hurricane' || getMove(ctx.battle, m).type === 'Water'; })) syn = true;
			if (wk === 'sun' && me.moves.some(function (m) { return m === 'solarbeam' || getMove(ctx.battle, m).type === 'Fire'; })) syn = true;
			if ((wk === 'snow' || wk === 'hail') && me.moves.some(function (m) { return m === 'blizzard' || m === 'auroraveil'; })) syn = true;
			return mix(wrap(plain() + (syn ? 0.12 : -0.2)));
		}

		if (id === 'healbell' || id === 'aromatherapy') {
			var sick = (me.status ? 1 : 0) + ((ctx.req.side.pokemon || []).filter(function (q) { return !q.active && / (par|brn|psn|tox|slp|frz)$/.test(q.condition); }).length);
			if (!sick) return FAIL;
			return mix(wrap(plain() + 0.1 * sick));
		}

		if (id === 'healingwish' || id === 'lunardance' || id === 'memento' || id === 'perishsong') return {V: -1};

		// Anything unrecognised is treated as a wasted turn.
		return {V: wrap(plain()) - 0.3};
	}

	// What changes the moment `X` comes in: hazard damage, poison from Toxic Spikes, the Speed drop
	// from Sticky Web, and its own Intimidate on whatever is across from it.
	function entryOf(ctx, X, opp) {
		var sc = ctx.field.mySide || {};
		var out = {X: X, opp: opp, dmg: hazardDmg(ctx, X)};
		if (X.item !== 'heavydutyboots' && isGrounded(ctx.field, X)) {
			if (sc.toxicspikes && !X.status && X.types.indexOf('Poison') < 0 && X.types.indexOf('Steel') < 0 && !hasAb(X, 'immunity', 'poisonheal', 'magicguard', 'pastelveil', 'comatose')) {
				out.X = clone(out.X, {status: (sc.toxicspikes[1] || 1) >= 2 ? 'tox' : 'psn', toxicTurns: 0});
			}
			if (sc.stickyweb && X.item !== 'clearamulet' && !hasAb(X, 'clearbody', 'whitesmoke', 'fullmetalbody')) {
				out.X = hasAb(X, 'defiant') ? withBoosts(out.X, {spe: -1, atk: 2}, false) : hasAb(X, 'competitive') ? withBoosts(out.X, {spe: -1, spa: 2}, false) : withBoosts(out.X, {spe: -1}, false);
			}
		}
		if (opp && hasAb(X, 'intimidate') && opp.item !== 'clearamulet' && !(opp.volatiles || {}).substitute) {
			if (abChance(opp, ['clearbody', 'whitesmoke', 'hypercutter', 'fullmetalbody', 'innerfocus', 'oblivious', 'owntempo', 'scrappy', 'mirrorarmor']) < 0.5) {
				out.opp = abChance(opp, ['defiant']) >= 0.5 ? withBoosts(opp, {atk: 1}, false) : abChance(opp, ['competitive']) >= 0.5 ? withBoosts(opp, {atk: -1, spa: 2}, false) :
					abChance(opp, ['guarddog']) >= 0.5 ? withBoosts(opp, {atk: 1}, false) : withBoosts(opp, {atk: -1}, false);
			}
		}
		return out;
	}

	function evalSwitch(S, X, st) {
		var ctx = S.ctx;
		// The incoming Pokemon takes either the move they picked for what is on the field now,
		// or, if they read the switch, their best move against the one coming in.
		var en = entryOf(ctx, X, S.opp);
		// Going down to the hazards on the way in gives the Pokemon away for nothing.
		if (en.X.hp - en.dmg <= 0) return {V: -en.X.hp / en.X.maxhp - KO_BONUS - 1.5, doa: true};
		var thrX = threat(ctx, en.opp, en.X);
		var hits = [];
		if (S.thr.best) hits.push({mv: S.thr.best.mv, p: 0.6 * S.thr.best.w});
		if (thrX.best) {
			if (hits.length && hits[0].mv.id === thrX.best.mv.id) hits[0].p = Math.min(1, hits[0].p + 0.4 * thrX.best.w);
			else hits.push({mv: thrX.best.mv, p: 0.4 * thrX.best.w});
		}
		var V;
		if (cfg.entryRisk) {
			V = entryValue(S, X, S.opp.hp, hits, 0.5 * thrX.exp);
			// Instead of attacking, a Pokemon with a setup move often uses the turn to boost. Then
			// nothing lands on the way in, and the one coming in faces it a stage or two up.
			var su = cfg.freeSetup > 0 ? setupOf(ctx, en.opp) : null;
			if (su) {
				var had = 0;
				for (var bk in su.boosts) if (su.boosts[bk] > 0 && (en.opp.boosts[bk] || 0) > 0) had++;
				var pS = clamp(su.w * cfg.freeSetup * (had ? 0.5 : 1), 0, 0.6);
				var hpS = en.X.hp - en.dmg;
				if (pS >= 0.05 && S.opp.hp > S.opp.maxhp * su.cost + 1) {
					var oppUp = withHp(withBoosts(en.opp, su.boosts, en.opp.item === 'whiteherb'), Math.max(1, S.opp.hp - S.opp.maxhp * su.cost));
					var vS = -en.dmg / en.X.maxhp + su.cost + race(ctx, en.X, oppUp, hpS, oppUp.hp, {ignoreLock: true}).score;
					V = (1 - pS) * V + pS * vS;
				}
			}
		} else {
			var lost = en.dmg + thrX.exp * 0.4;
			if (S.thr.best) lost += calc(ctx, en.opp, en.X, S.thr.best.mv, {}).exp * S.thr.best.w * 0.6;
			var hpIn = en.X.hp - lost;
			V = hpIn <= 0 ? -en.X.hp / en.X.maxhp - KO_BONUS : -lost / en.X.maxhp + race(ctx, en.X, en.opp, hpIn, S.opp.hp, {ignoreLock: true}).score;
		}
		V -= cfg.switchPenalty + 0.1 * (st.switchStreak || 0);
		var lostIn = en.dmg;
		hits.forEach(function (h) { if (h.mv) lostIn += h.p * calc(ctx, en.opp, en.X, h.mv, {}).exp; });
		lostIn = clamp(lostIn / en.X.maxhp, 0, en.X.hp / en.X.maxhp);
		// Do not bounce straight back to something that just left.
		var left = (st.leftAt || {})[X.ident];
		if (left && S.turn - left <= 3) V -= 0.25;
		// Boosts are lost on the way out.
		var b = S.me.boosts || {}, up = 0;
		['atk', 'spa', 'spe', 'def', 'spd'].forEach(function (k) { if (b[k] > 0) up += b[k]; });
		V -= Math.min(0.4, 0.07 * up);
		if (hasAb(S.me, 'regenerator')) V += 0.1;
		return {V: V, lostIn: lostIn};
	}

	// ------------------------------------------------------------------ whole-team awareness

	// The opponent's revealed, healthy benched Pokemon.
	function oppBenchList(ctx) {
		var far = ctx.battle.farSide;
		var out = [];
		if (!far) return out;
		var active = far.active ? far.active[0] : null;
		(far.pokemon || []).forEach(function (q) {
			if (q === active || q.fainted || !(q.hp > 0)) return;
			try { out.push(mkOpp(ctx, q)); } catch (e) {}
		});
		return out;
	}

	// Share of recent turns in which the opponent chose to switch.
	function oppSwitchRate(st, turn) {
		var span = Math.min(5, Math.max(0, turn - 1));
		if (span <= 0) return 0;
		var n = (st.oppSwitchTurns || []).filter(function (t) { return t >= turn - span && t < turn; }).length;
		return clamp(n / span, 0, 1);
	}

	// The benched Pokemon the opponent would most want to bring in against my active one, and how
	// likely it is to make that switch this turn.
	function likelySwitchIn(ctx, me, opp, benchOpp, st, turn) {
		if (!benchOpp.length) return null;
		var obvious = bestAttack(ctx, me, opp, {}); // the move they expect me to click
		var best = null;
		benchOpp.forEach(function (Y) {
			var haz = hazardDmg(ctx, Y);
			var hp = Y.hp - haz;
			if (hp <= 0) return;
			var sc = race(ctx, me, Y, me.hp, hp).score;
			// A switch-in has to take the move I am expected to click, so players reach for
			// whatever shrugs that move off, ahead of whatever wins the matchup on paper.
			var eat = obvious.mv ? Math.min(1, calc(ctx, me, withHp(Y, hp), obvious.mv, {firstTurn: false}).exp / Y.maxhp) : 0;
			var pick = sc + cfg.switchEat * eat;
			if (!best || pick < best.pick) best = {Y: Y, haz: haz, hp: hp, score: sc, pick: pick, eat: eat};
		});
		if (!best) return null;
		var stayR = race(ctx, me, opp, me.hp, opp.hp);
		var stay = stayR.score;
		// How badly they want out of this matchup, and how much the bench improves on it.
		var pressure = clamp((stay - 0.1) / 0.6, 0, 1);
		var relief = clamp((stay - best.score) / 0.5, 0, 1);
		// Some opponents leave a bad matchup every time and some never do. Count what this one
		// did on the earlier turns it was in trouble, starting from "one time in three".
		var seen = st.pickle || {n: 0, sw: 0};
		var habit = (seen.sw + 0.5) / (seen.n + 1.5) * 3;
		var w = cfg.predictSwitch + 0.4 * oppSwitchRate(st, turn) + cfg.switchPressure * pressure * relief * habit;
		// Locked into a move that barely scratches me: staying in is not really an option.
		if (lockedInto(opp) && stayR.thr.exp < 0.12 * me.maxhp) w += 0.25 * relief;
		// They take the KO before I move: no reason for them to leave.
		if (stayR.thr.best && stayR.f < 0.5 && koProb(stayR.thr.best.r, me.hp, me) >= 0.7) w *= 0.3;
		if (best.score >= stay - 0.1) w *= 0.3; // their bench does no better than what is already in
		if (cfg.swCal) {
			// From 5573 turns of ladder games: with a KO threatened they left 7 points more often
			// than weighted, with lowered stats about 15 more, and the highest weights ran 5 over.
			if (obvious.r && koProb(obvious.r, opp.hp, opp) >= 0.5) w += 0.07;
			var dropped = false, raised = false;
			for (var sk in opp.boosts) { if (opp.boosts[sk] < 0) dropped = true; else if (opp.boosts[sk] > 0) raised = true; }
			if (dropped && !raised) w += 0.15;
			if (w > 0.36) w = 0.36 + 0.5 * (w - 0.36);
		}
		best.w = clamp(w, 0, cfg.predictCap);
		best.trouble = pressure * relief;
		return best;
	}

	// What an option is worth if the opponent switches to `sw.Y` instead of attacking.
	function valueVsSwitch(S, o) {
		var ctx = S.ctx, me = S.me, sw = S.sw, Y = sw.Y;
		var hazGain = sw.haz / Y.maxhp;
		var hpY = sw.hp;
		if (o.V <= -2.5) return o.V;
		if (o.isSwitch) {
			var enX = entryOf(ctx, o.X, Y);
			var X = enX.X, hazX = enX.dmg;
			if (X.hp - hazX <= 0) return -X.hp / X.maxhp - KO_BONUS;
			return hazGain - hazX / X.maxhp + race(ctx, X, enX.opp, X.hp - hazX, hpY, {ignoreLock: true}).score - cfg.switchPenalty;
		}
		var mv = o.mv;
		var Yin = withHp(Y, hpY);
		if (o.attack) {
			var r = calc(ctx, me, Yin, mv, {firstTurn: S.firstTurn, fallen: S.fallen});
			var act = actChance(ctx, me);
			var dealt = Math.min(hpY, r.exp) * act;
			var myFrac = me.hp / me.maxhp;
			if (mv.selfdestruct || mv.id === 'finalgambit') return hazGain + dealt / Y.maxhp - myFrac - KO_BONUS;
			var selfDmg = 0;
			if (mv.recoil && !hasAb(me, 'rockhead', 'magicguard')) selfDmg += dealt * mv.recoil[0] / mv.recoil[1];
			if (me.item === 'lifeorb' && dealt > 0 && !hasAb(me, 'magicguard', 'sheerforce')) selfDmg += me.maxhp / 10;
			var hpMe = me.hp - selfDmg;
			if (hpMe <= 0) return hazGain + dealt / Y.maxhp - myFrac - KO_BONUS;
			var pk = koProb(r, hpY, Yin) * act;
			var hpYAfter = Math.max(1, hpY - dealt);
			var contV;
			if (mv.selfSwitch && S.bench.length) {
				contV = -2;
				S.bench.forEach(function (X2) {
					var hz = hazardDmg(ctx, X2);
					var v2 = X2.hp - hz <= 0 ? -X2.hp / X2.maxhp - KO_BONUS : -hz / X2.maxhp + race(ctx, X2, Y, X2.hp - hz, hpYAfter, {ignoreLock: true}).score;
					if (v2 > contV) contV = v2;
				});
			} else {
				var me2 = withHp(me, hpMe);
				if (mv.self && mv.self.boosts) {
					me2.boosts = Object.assign({}, me.boosts);
					for (var k in mv.self.boosts) me2.boosts[k] = clamp((me2.boosts[k] || 0) + mv.self.boosts[k], -6, 6);
				}
				contV = race(ctx, me2, Y, hpMe, hpYAfter).score;
			}
			return hazGain + dealt / Y.maxhp - selfDmg / me.maxhp + pk * KO_BONUS + (1 - pk) * contV;
		}
		// Status moves: the opponent's switch gives me the turn for free.
		var id = mv.id, f = ctx.field;
		var base = function (me3, Y3, hp3) { return hazGain + race(ctx, me3, Y3 || Y, hp3 === undefined ? me.hp : hp3, hpY).score; };
		if (PROTECT_MOVES[id] || mv.stallingMove) return base(me) - 0.08;
		var boosts = id === 'bellydrum' ? {atk: 6} : mv.boosts;
		if (boosts && (mv.target === 'self' || id === 'bellydrum')) {
			var meB = clone(me, {boosts: Object.assign({}, me.boosts)});
			for (var bk in boosts) meB.boosts[bk] = clamp((meB.boosts[bk] || 0) + boosts[bk], -6, 6);
			var cost = id === 'bellydrum' || id === 'filletaway' ? me.maxhp / 2 : id === 'clangoroussoul' ? me.maxhp / 3 : 0;
			return base(meB, Y, me.hp - cost) - cost / me.maxhp;
		}
		if ((mv.heal || mv.flags.heal || id === 'rest') && mv.target !== 'normal') {
			var amt = mv.heal ? me.maxhp * mv.heal[0] / mv.heal[1] : id === 'rest' ? me.maxhp : me.maxhp / 2;
			if (id === 'moonlight' || id === 'synthesis' || id === 'morningsun') amt = me.maxhp * (f.sun ? 2 / 3 : f.weather ? 1 / 4 : 1 / 2);
			var healed = Math.min(me.maxhp, me.hp + amt);
			return (healed - me.hp) / me.maxhp + base(me, Y, healed);
		}
		if (mv.sideCondition && HAZARD_VALUE[mv.sideCondition]) {
			var cur = f.oppSide[mv.sideCondition];
			var layers = cur ? (cur[1] || 1) : 0;
			return base(me) + HAZARD_VALUE[mv.sideCondition] * Math.min(4, Math.max(0, S.oppAlive - 1)) * (layers ? 0.6 : 1);
		}
		if ((mv.status || id === 'yawn') && mv.target !== 'self') {
			var st2 = mv.status || 'slp';
			var land = statusLands(ctx, me, Yin, mv) * (id === 'yawn' ? 0.7 : 1);
			var Ys = clone(Yin, {status: st2, toxicTurns: 0});
			var extra = st2 === 'slp' ? {myFreeHits: f.gen === 1 ? 3 : 1} : {};
			return hazGain + land * race(ctx, me, Ys, me.hp, hpY, extra).score + (1 - land) * race(ctx, me, Y, me.hp, hpY).score;
		}
		return base(me) - 0.1;
	}

	// In three losses out of four one opposing Pokemon takes three or more KOs, and more often
	// than not it had been seen well before it started, while the bot was ahead. So look at what
	// is waiting on their bench: a Pokemon there that beats two or more of mine, with only one
	// or two of mine able to beat it, makes those one or two worth keeping healthy. Returns, by
	// team slot, how much of that weight each of my Pokemon carries.
	function answersTo(ctx, me, bench, benchOpp) {
		var out = {};
		var mineAll = [me].concat(bench || []);
		if (mineAll.length < 2 || !benchOpp || !benchOpp.length) return out;
		benchOpp.forEach(function (Y) {
			if (!(Y.hp > 0)) return;
			var winners = [], beaten = 0;
			mineAll.forEach(function (X) {
				var sc = race(ctx, X, Y, X.hp, Y.hp, {ignoreLock: true}).score;
				if (sc >= 0.15) winners.push(X); else if (sc <= -0.3) beaten++;
			});
			if (beaten < 2 || !winners.length || winners.length > 2) return;
			var w = clamp(beaten / 3, 0, 1) / winners.length;
			winners.forEach(function (X) { out[X.slot] = (out[X.slot] || 0) + w; });
		});
		return out;
	}

	// Blend the value against the opponent staying in with the value against its likely switch.
	function withSwitchRisk(S, o) {
		if (!S.sw || !S.sw.w) return o.V;
		var vs = valueVsSwitch(S, o);
		o.Vstay = o.V;
		o.Vsw = vs;
		return (1 - S.sw.w) * o.V + S.sw.w * vs;
	}

	// ------------------------------------------------------------------ decision

	function makeCtx(room, req) {
		var battle = room.battle;
		var gen = battle.gen || 9;
		var fmt = (room.id || '').split('-')[1] || toID(battle.tier);
		loadSets(fmt, gen);
		return {battle: battle, room: room, req: req, gen: gen, fmt: fmt, field: mkField(battle), term: calibrate(battle, req)};
	}

	function generalValue(ctx, X) {
		var foes = ((ctx.battle.farSide && ctx.battle.farSide.pokemon) || []).filter(function (q) { return !q.fainted && q.hp > 0; });
		var v = 0.5 * X.hp / X.maxhp;
		if (!foes.length) return v;
		var sum = 0;
		foes.forEach(function (q) {
			try {
				var o = mkOpp(ctx, q);
				sum += race(ctx, X, o, X.hp, o.hp, {ignoreLock: true}).score;
			} catch (e) {}
		});
		return v + 0.5 * sum / foes.length;
	}

	function megaSuffix(act, me) {
		if (act.canMegaEvo) return ' mega';
		if (act.canMegaEvoX && act.canMegaEvoY) return (me.stats.atk >= me.stats.spa) ? ' megax' : ' megay';
		if (act.canMegaEvoX) return ' megax';
		if (act.canMegaEvoY) return ' megay';
		if (act.canUltraBurst) return ' ultra';
		return '';
	}

	function decide(room, req, st) {
		var base = st && st.arm === 'base' && cfg.arms && cfg.arms.base;
		if (!base) return decideWith(room, req, st);
		var saved = {};
		for (var k in base) { saved[k] = cfg[k]; cfg[k] = base[k]; }
		try { return decideWith(room, req, st); } finally { for (var k2 in saved) cfg[k2] = saved[k2]; }
	}

	function decideWith(room, req, st) {
		var battle = room.battle;
		if (req.teamPreview) return {choice: 'default', why: 'team preview', table: []};
		if (!req.side || !req.side.pokemon) return {choice: 'default', why: 'no side data', table: []};
		var ctx = makeCtx(room, req);
		ctx.st = st;
		// Did the opponent leave the bad matchup I saw last time? That sets how much weight
		// the next prediction gets.
		if (st.predicted && st.predicted.turn < battle.turn) {
			predictions.made++;
			predictions.weight += st.predicted.w;
			if ((st.oppSwitchTurns || []).indexOf(st.predicted.turn) >= 0) {
				predictions.switched++;
				if ((st.oppSwitchIns || {})[st.predicted.turn] === toID(st.predicted.name)) predictions.rightMon++;
			}
			st.predicted = null;
		}
		if (st.pickleTurn !== undefined && st.pickleTurn < battle.turn) {
			st.pickle = st.pickle || {n: 0, sw: 0};
			st.pickle.n++;
			if ((st.oppSwitchTurns || []).indexOf(st.pickleTurn) >= 0) st.pickle.sw++;
			st.pickleTurn = undefined;
		}
		var mons = req.side.pokemon;
		var activeIdx = -1;
		mons.forEach(function (p, i) { if (p.active && activeIdx < 0) activeIdx = i; });
		var far = battle.farSide;
		var oppCm = far && far.active ? far.active[0] : null;
		var opp = oppCm && !oppCm.fainted && oppCm.hp > 0 ? mkOpp(ctx, oppCm) : null;
		var bad = st.badChoices || {};
		var bench = [];
		mons.forEach(function (p, i) {
			if (p.active) return;
			var c = parseCond(p.condition);
			if (c.fainted) return;
			if (bad['switch ' + (i + 1)]) return;
			bench.push(mkMine(ctx, p, null, i + 1));
		});
		var oppFainted = far ? far.pokemon.filter(function (q) { return q.fainted; }).length : 0;
		var oppAlive = far ? Math.max(1, (far.totalPokemon || 6) - oppFainted) : 6;
		var fallen = mons.filter(function (p) { return parseCond(p.condition).fainted; }).length;
		var benchOpp = oppBenchList(ctx);

		// ---------- forced switch
		if (req.forceSwitch) {
			if (activeIdx >= 0 && mons[activeIdx].reviving) {
				for (var ri = 0; ri < mons.length; ri++) {
					if (parseCond(mons[ri].condition).fainted && !bad['switch ' + (ri + 1)]) return {choice: 'switch ' + (ri + 1), why: 'revive', table: []};
				}
			}
			if (!bench.length) return {choice: 'default', why: 'nothing to switch to', table: []};
			// Is this the switch after my own U-turn, with their move still to come this turn? Then
			// whoever comes in takes the move they picked for the Pokemon that is leaving.
			var pivotHits = null, pivotS = null;
			var leaving = activeIdx >= 0 ? mons[activeIdx] : null;
			if (opp && leaving && !parseCond(leaving.condition).fainted && !st.oppMoved && st.turnNow === battle.turn) {
				try {
					var meOut = mkMine(ctx, leaving, null, activeIdx + 1);
					var thrOut = threat(ctx, opp, meOut);
					if (thrOut.best) { pivotHits = [{mv: thrOut.best.mv, p: thrOut.best.w}]; pivotS = {ctx: ctx, opp: opp}; }
				} catch (e) {}
			}
			var sopts = bench.map(function (X0) {
				var v;
				var en = entryOf(ctx, X0, opp);
				var X = en.X;
				if (pivotHits) {
					v = entryValue(pivotS, X0, opp.hp, pivotHits, 0);
				} else if (opp) {
					var hp = X.hp - en.dmg;
					v = hp <= 0 ? -X.hp / X.maxhp - KO_BONUS : -(X.hp - hp) / X.maxhp + race(ctx, X, en.opp, hp, opp.hp, {ignoreLock: true}).score;
				} else {
					v = generalValue(ctx, X);
				}
				if (opp && benchOpp.length) {
					var sum = 0;
					benchOpp.forEach(function (Y) { sum += race(ctx, X, Y, Math.max(1, X.hp - en.dmg), Y.hp, {ignoreLock: true}).score; });
					v = 0.75 * v + 0.25 * sum / benchOpp.length;
				}
				return {choice: 'switch ' + X0.slot, label: 'switch ' + X0.name, V: v, isSwitch: true, doa: X.hp - en.dmg <= 0};
			});
			// One that goes down to the hazards on the way in is lost for nothing, and the choice
			// comes straight back. It goes last.
			if (sopts.some(function (o) { return !o.doa; })) sopts.forEach(function (o) { if (o.doa) o.V -= 1.5; });
			sopts.sort(function (a, b) { return b.V - a.V; });
			return {choice: sopts[0].choice, why: sopts[0].label + (opp ? ' vs ' + opp.name : ''), table: sopts};
		}

		if (!req.active || !req.active[0]) return {choice: 'default', why: 'no active request', table: []};
		if (req.active.length > 1 || battle.gameType !== 'singles') return {choice: 'default', why: 'only singles is supported', table: []};

		var act = req.active[0];
		var myCm = battle.nearSide && battle.nearSide.active ? battle.nearSide.active[0] : null;
		var me = mkMine(ctx, mons[activeIdx >= 0 ? activeIdx : 0], myCm, activeIdx + 1);
		if (st.activeIdent !== mons[activeIdx >= 0 ? activeIdx : 0].ident) {
			st.activeIdent = mons[activeIdx >= 0 ? activeIdx : 0].ident;
			st.enteredTurn = battle.turn;
		}
		var firstTurn = st.enteredTurn === battle.turn;

		var usable = [];
		(act.moves || []).forEach(function (m, i) {
			if (m.disabled) return;
			if (m.pp !== undefined && m.pp <= 0) return;
			if (bad['move ' + (i + 1)]) return;
			usable.push({slot: i + 1, id: toID(m.id || m.move)});
		});
		var mega = megaSuffix(act, me);
		var canSwitch = !act.trapped && bench.length > 0 && st.noSwitchRqid !== req.rqid;
		if (!usable.length) return {choice: canSwitch && bad['move 1'] ? 'switch ' + bench[0].slot : 'move 1', why: 'no usable move', table: []};
		if (!opp) return {choice: 'move ' + usable[0].slot + mega, why: 'no opposing Pokemon on the field', table: []};
		if (usable.length === 1 && !canSwitch) return {choice: 'move ' + usable[0].slot + mega, why: 'only option', table: []};

		var evalAll = function (meX, tag, suffix, alt) {
			alt = alt || {};
			var oppX = alt.opp || opp;
			var S = {ctx: ctx, st: st, me: meX, opp: oppX, thr: alt.thr || threat(ctx, oppX, meX), bench: bench, oppAlive: oppAlive, firstTurn: firstTurn, fallen: fallen, turn: battle.turn, benchOpp: benchOpp, sw: likelySwitchIn(ctx, meX, oppX, benchOpp, st, battle.turn)};
			S.oppHeal = healOf(ctx, oppX);
			S.oppSetup = setupOf(ctx, oppX);
			S.sucker = suckerRel(S);
			S.prioKo = 0;
			usable.forEach(function (u) {
				var pm = getMove(battle, u.id);
				if (pm.category === 'Status' || movePriority(ctx, meX, pm) <= 0) return;
				S.prioKo = Math.max(S.prioKo, koProb(calc(ctx, meX, oppX, pm, {firstTurn: firstTurn}), oppX.hp, oppX));
			});
			var evals = usable.map(function (u) {
				var mv = getMove(battle, u.id);
				if (meX.dynamaxed) mv = asMax(mv);
				return {u: u, mv: mv, e: mv.category === 'Status' ? evalStatus(S, {mv: mv}) : evalAttack(S, {mv: mv})};
			});
			// Protect is worth the best of the other moves, a turn later.
			var bestOther = null;
			evals.forEach(function (x) { if (x.e.protect === undefined && !x.e.fail && (bestOther === null || x.e.V > bestOther)) bestOther = x.e.V; });
			evals.forEach(function (x) { if (x.e.protect !== undefined && !x.e.fail && bestOther !== null) x.e.V = bestOther + x.e.protect; });
			var outs = evals.map(function (x) {
				var u = x.u, mv = x.mv, e = x.e;
				var o = {choice: 'move ' + u.slot + mega + (suffix || ''), label: mv.name + (tag ? ' +' + tag : ''), V: e.V, pKO: e.pKO, pDead: e.pDead, dealt: e.dealt, slot: u.slot, mv: mv, attack: mv.category !== 'Status', S: S,
					setup: !!e.setup, heal: !!e.heal, suicide: !!e.suicide, r: e.r || null};
				o.V = withSwitchRisk(S, o);
				// A KO that is all but certain, taken without much risk, is worth more than the
				// HP it removes: nothing the opponent planned for that Pokemon happens.
				if (o.attack && (o.pKO || 0) >= 0.9 && (o.pDead || 0) <= 0.15) o.V += cfg.sureKo;
				// While the opponent is setting up, a turn that does nothing about it is a turn
				// it gets stronger.
				if (settingUp && !o.attack && !BOOST_ANSWERS[mv.id] && !mv.status && mv.id !== 'taunt') o.V -= cfg.urgency;
				return o;
			});
			// Sleep Talk picks one of the other moves at random.
			if (meX.status === 'slp') {
				outs.forEach(function (o) {
					if (o.mv.id !== 'sleeptalk') return;
					var others = usable.filter(function (u) { return u.id !== 'sleeptalk'; }).map(function (u) {
						var mv2 = getMove(battle, u.id);
						var awake = clone(meX, {status: ''});
						var S2 = clone(S, {me: awake});
						return (mv2.category === 'Status' ? evalStatus(S2, {mv: mv2}) : evalAttack(S2, {mv: mv2})).V;
					});
					if (others.length) o.V = others.reduce(function (x, y) { return x + Math.max(y, -1); }, 0) / others.length;
				});
			}
			return outs;
		};

		// Is the opponent in the middle of setting up? Its last move was a boost, or it already
		// carries two or more stages.
		var lastOpp = opp.lastMove ? getMove(battle, opp.lastMove) : null;
		var stages = Math.max(0, opp.boosts.atk || 0) + Math.max(0, opp.boosts.spa || 0) + Math.max(0, opp.boosts.spe || 0);
		var settingUp = cfg.urgency > 0 && ((lastOpp && lastOpp.category === 'Status' && lastOpp.target === 'self' && !!lastOpp.boosts) || stages >= 2);

		// Scores every usable move, then folds in the chance that the opponent Terastallizes
		// this turn into one of the types its set allows.
		var teraGuess = oppTeraGuess(ctx, me, opp);
		var evalMoves = function (meX, tag, suffix, alt) {
			var outs = evalAll(meX, tag, suffix, alt);
			if (!teraGuess) return outs;
			var alts = teraGuess.forms.map(function (oppT) { return evalAll(meX, tag, suffix, {opp: oppT}); });
			outs.forEach(function (o, i) {
				var sum = 0;
				alts.forEach(function (tb) { sum += tb[i].V; });
				o.Vplain = o.V;
				o.V = (1 - teraGuess.p) * o.V + teraGuess.p * sum / alts.length;
			});
			return outs;
		};
		var table = evalMoves(me, '', '');
		var S0 = table.length ? table[0].S : null;

		// HP spent by the Pokemon that answers a threat on their bench costs more than its share.
		var keep = cfg.keepAnswers > 0 && S0 ? answersTo(ctx, me, bench, benchOpp) : {};
		var keepMe = keep[me.slot] || 0;
		var fNowK = 1, takeK = 0;
		if (keepMe > 0) {
			fNowK = S0.thr.best ? firstProb(ctx, me, opp, 0, S0.thr.best.r.priority) : 1;
			takeK = Math.min(me.hp, S0.thr.exp) / me.maxhp;
			table.forEach(function (o) {
				var pd = o.pDead || 0;
				var lossFrac = pd * me.hp / me.maxhp + (1 - pd) * takeK * (1 - (o.pKO || 0) * fNowK);
				o.keepCost = cfg.keepAnswers * Math.min(1, keepMe) * lossFrac;
				o.V -= o.keepCost;
			});
		}
		if (canSwitch && S0) {
			bench.forEach(function (X) {
				var e = evalSwitch(S0, X, st);
				var so = {choice: 'switch ' + X.slot, label: 'switch ' + X.name, V: e.V, isSwitch: true, X: X};
				so.V = withSwitchRisk(S0, so);
				if (keep[X.slot] && e.lostIn) { so.keepCost = cfg.keepAnswers * Math.min(1, keep[X.slot]) * e.lostIn; so.V -= so.keepCost; }
				table.push(so);
			});
		}
		table.sort(function (a, b) { return b.V - a.V; });
		// Switching hands the opponent a free turn, so only do it to escape a matchup that is
		// actually going badly, never just because a teammate looks slightly better.
		var bestStay = null;
		table.forEach(function (o) { if (!o.isSwitch && !bestStay) bestStay = o; });
		var best = table[0];
		// The exception is a Pokemon about to go down without taking anything with it: then any
		// clearly better teammate is worth the switch.
		var doomed = bestStay && (bestStay.pDead || 0) >= 0.6 && (bestStay.pKO || 0) < 0.5;
		// The more staying in is still worth, the bigger the gain a switch has to show.
		var needed = cfg.switchMargin + cfg.switchStayTax * clamp(bestStay ? bestStay.V : 0, 0, 0.6) - cfg.switchRelief * clamp(bestStay ? -bestStay.V / 0.35 : 0, 0, 1);
		if (doomed) needed = Math.min(needed, cfg.doomedMargin);
		// Asleep or frozen with no way to act: every turn spent here is a free turn for them.
		var helpless = actChance(ctx, me) <= 0.34 && opp.status !== 'slp' && opp.status !== 'frz' && !usable.some(function (u) { return u.id === 'sleeptalk'; });
		if (helpless) needed = Math.min(needed, cfg.sleepMargin);
		// A switch is scored together with the matchup that follows it. Going down in place is
		// followed by a matchup too: the next Pokemon comes in without taking a hit. Count that
		// for the one that is about to go down, or a healthy teammate gets thrown into the hit
		// to save a Pokemon that is already lost.
		var stayV = bestStay ? bestStay.V : 0;
		if (best.isSwitch && bestStay && S0 && cfg.revenge > 0 && (bestStay.pDead || 0) >= 0.4) stayV += (bestStay.pDead || 0) * Math.max(0, revengeOf(S0));
		if (best.isSwitch && bestStay && (bestStay.V >= cfg.switchOnlyBelow || best.V - stayV < needed)) best = bestStay;

		// A KO that is all but certain and safe to take comes first: over healing, over another
		// boost, over a switch. The one exception is setting up on something that cannot hurt me.
		var sureKo = function (o) {
			return o && o.attack && !o.isSwitch && (o.pKO || 0) >= 0.85 && (o.pDead || 0) <= 0.3 && !o.suicide &&
				!(o.mv && (o.mv.selfdestruct || o.mv.id === 'finalgambit'));
		};
		var tookKo = false;
		if (cfg.takeKo && S0) {
			var kos = table.filter(sureKo);
			if (kos.length) {
				// Of the KOs that score about the same, the one most certain to land.
				var pick = kos[0];
				kos.forEach(function (o) { if (o.V >= kos[0].V - 0.1 && (o.pKO || 0) > (pick.pKO || 0) + 0.04) pick = o; });
				if (!sureKo(best)) {
					var harmless = S0.thr.exp <= 0.1 * me.maxhp && !S0.oppSetup && !S0.oppHeal;
					var freeSetup = !best.isSwitch && best.setup && harmless && oppAlive > 1;
					if (!freeSetup) { best = pick; tookKo = true; }
				} else if (pick !== best && best.V - pick.V <= 0.1 && (pick.pKO || 0) > (best.pKO || 0) + 0.04) {
					best = pick;
				}
			}
		}

		// ---------- Terastallize
		var teraWhy = '';
		if (act.canTerastallize && !me.tera) {
			var tt = act.canTerastallize;
			var meT = clone(me, {tera: tt, types: tt === 'Stellar' ? me.types : [tt]});
			meT.grounded = isGrounded(ctx.field, meT);
			// The opponent chose its move without knowing about the Tera, so score this turn
			// against the move it picked for my old typing.
			var tTable = evalMoves(meT, 'Tera ' + tt, ' terastallize', cfg.teraSurprise ? {thr: surpriseThreat(ctx, opp, me, meT)} : null);
			if (!cfg.teraStatus) tTable = tTable.filter(function (o) { return o.attack; });
			// The same extra cost of HP for a Pokemon that answers a benched threat applies with the Tera.
			if (keepMe > 0) {
				tTable.forEach(function (o) {
					var pd = o.pDead || 0;
					o.keepCost = cfg.keepAnswers * Math.min(1, keepMe) * (pd * me.hp / me.maxhp + (1 - pd) * takeK * (1 - (o.pKO || 0) * fNowK));
					o.V -= o.keepCost;
				});
			}
			tTable.sort(function (a, b) { return b.V - a.V; });
			var bestT = tTable[0];
			if (bestT) {
				var myAlive = bench.length + 1;
				// Tera is spent once. The best players hold it until the game has taken shape:
				// around turn 19 with two of their own Pokemon down, almost never in the first two
				// turns. So the bar starts high and comes down as Pokemon fall.
				var early = clamp(1 - (fallen + oppFainted) / 4, 0, 1);
				var late = myAlive <= 2 || oppAlive <= 2;
				var margin = late ? cfg.teraMargin * 0.4 : cfg.teraMargin + 2 * cfg.teraEarly * early + (battle.turn <= 2 ? 0.3 : 0);
				var koGain = late ? 0.08 : cfg.teraKoGain + 0.35 * early;
				// The gain disappears if they switch out of the hit, and the Tera is gone anyway.
				if (S0 && S0.sw && S0.sw.w >= 0.25 && !late) { margin += 0.3 * S0.sw.w; koGain += 0.3 * S0.sw.w; }
				var gain = bestT.V - best.V;
				var plain = best.isSwitch ? bestStay : best; // what staying in looks like without it
				// Two cases are worth the Tera on their own: it turns a hit into a KO, or it turns
				// a KO on me into a hit I survive.
				var securesKO = bestT.attack && (bestT.pKO || 0) >= 0.85 && (bestT.pDead || 0) <= 0.3 && (!plain || (plain.pKO || 0) < 0.5) && !tookKo;
				var dodgesKO = cfg.teraSurprise && !!plain && (plain.pDead || 0) >= 0.5 && (bestT.pDead || 0) <= 0.2;
				// Not while asleep, and not into a Substitute that eats the hit.
				var wasted = actChance(ctx, me) <= 0.34 || ((opp.volatiles || {}).substitute && !(bestT.mv && bestT.mv.flags.sound));
				// It keeps alive a Pokemon that was about to go down, and the matchup is better for it.
				var plainAtk = null;
				table.forEach(function (o) { if (o.attack && !o.isSwitch && !plainAtk) plainAtk = o; });
				var savesMe = !!plainAtk && (plainAtk.pDead || 0) >= 0.5 && (bestT.pDead || 0) <= 0.2;
				if (!wasted && !tookKo && (gain >= margin || ((securesKO || dodgesKO) && gain >= koGain) || (cfg.teraSave && savesMe && gain >= koGain + 0.1 + (battle.turn <= 2 ? 0.3 : 0)))) {
					best = bestT;
					if (cfg.takeKo && !sureKo(bestT) && !bestT.setup) {
						var tKos = tTable.filter(sureKo);
						if (tKos.length) {
							best = tKos[0];
							tKos.forEach(function (o) { if (o.V >= tKos[0].V - 0.1 && (o.pKO || 0) > (best.pKO || 0) + 0.04) best = o; });
						}
					}
					teraWhy = securesKO || best !== bestT ? 'Tera secures the KO' : dodgesKO || savesMe ? 'Tera survives the hit' : 'Tera wins the trade';
				}
			}
			table = table.concat(tTable);
		}

		// ---------- Dynamax
		if (act.canDynamax && !me.dynamaxed) {
			var meD = clone(me, {dynamaxed: true, hp: me.hp * 2, maxhp: me.maxhp * 2});
			var dTable = evalAll(meD, 'Dynamax', ' dynamax').filter(function (o) { return o.attack; });
			dTable.sort(function (a, b) { return b.V - a.V; });
			var alive = bench.length + 1;
			var dMargin = alive <= 2 ? 0.05 : oppAlive <= 2 ? 0.12 : 0.3;
			if (dTable.length && dTable[0].V - best.V >= dMargin) best = dTable[0];
			table = table.concat(dTable);
		}

		// ---------- Z-Move: spend it when it turns a non-kill into a kill
		if (act.canZMove && best.attack && !best.isSwitch && act.canZMove[best.slot - 1]) {
			var zr = calc(ctx, me, opp, asZ(best.mv), {});
			var zk = koProb(zr, opp.hp, opp);
			if ((zk >= 0.7 && (best.pKO || 0) < 0.6) || bench.length === 0) best = clone(best, {choice: best.choice + ' zmove', label: best.label + ' +Z'});
		}

		table.sort(function (a, b) { return b.V - a.V; });
		var predict = S0 && S0.sw && S0.sw.w >= 0.3 ? {name: S0.sw.Y.name, w: S0.sw.w} : null;
		if (S0 && S0.sw && S0.sw.trouble >= 0.25) st.pickleTurn = battle.turn;
		if (predict) st.predicted = {turn: battle.turn, name: predict.name, w: predict.w};
		var notes = [];
		if (teraWhy && / terastallize/.test(best.choice)) notes.push(teraWhy);
		if (predict) notes.push('expecting a switch to ' + predict.name + ' (' + pct(predict.w) + ')');
		if (teraGuess && teraGuess.p >= 0.25) notes.push('wary of their Tera');
		if (tookKo) notes.push('takes the KO');
		var dbg = null;
		try {
			var fNow = S0 && S0.thr.best ? firstProb(ctx, me, opp, 0, S0.thr.best.r.priority) : 1;
			dbg = {
				me: {spe: Math.round(effSpeed(ctx, me)), boosts: me.boosts, item: me.item, ability: me.ability, tera: me.tera || '', canTera: act.canTerastallize || ''},
				opp: {spe: Math.round(effSpeed(ctx, opp)), boosts: opp.boosts, item: opp.itemKnown ? opp.item : '?' + Object.keys(opp.itemW || {}).join('/'), ability: opp.abilityKnown ? opp.ability : '?' + (opp.possibleAbilities || []).join('/'),
					moves: opp.moves, pool: opp.pool, tera: opp.tera || '', possTera: opp.possTera, choiceP: round3(opp.choiceP)},
				threat: S0 && S0.thr.best ? {move: S0.thr.best.mv.id, exp: Math.round(S0.thr.exp), min: S0.thr.best.r.min, max: S0.thr.best.r.max, w: round3(S0.thr.best.w)} : null,
				moveFirst: round3(fNow), heals: S0 && S0.oppHeal ? {id: S0.oppHeal.id, w: round3(S0.oppHeal.w)} : null,
				sets: S0 && S0.oppSetup ? {id: S0.oppSetup.id, w: round3(S0.oppSetup.w)} : null,
				sw: S0 && S0.sw ? {to: S0.sw.Y.name, w: round3(S0.sw.w)} : null, tera: teraGuess ? round3(teraGuess.p) : 0,
				gate: {needed: round3(needed), doomed: !!doomed, helpless: !!helpless, tookKo: tookKo}, trapped: !!act.trapped, canSwitch: canSwitch,
				keep: Object.keys(keep).length ? keep : null
			};
		} catch (e) {}
		return {choice: best.choice, why: best.label + ' vs ' + opp.name + (notes.length ? ', ' + notes.join(', ') : ''), table: table, best: best, predict: predict, teraWhy: teraWhy, dbg: dbg};
	}

	// ------------------------------------------------------------------ client glue

	function isSynced(room, req) {
		var battle = room.battle;
		if (!battle || battle.seeking !== null) return false;
		if (!battle.atQueueEnd) return false;
		if (req.teamPreview) return true;
		if (!req.side || !req.side.pokemon) return true;
		var mineP = null;
		req.side.pokemon.forEach(function (p) { if (p.active && !mineP) mineP = p; });
		var cm = battle.nearSide && battle.nearSide.active ? battle.nearSide.active[0] : null;
		if (req.forceSwitch) return true;
		if (!mineP || !cm) return false;
		var cond = parseCond(mineP.condition);
		if (cond.maxhp && cm.maxhp) {
			if (Math.abs(cond.hp / cond.maxhp - cm.hp / cm.maxhp) > 0.03) return false;
		}
		var oc = battle.farSide && battle.farSide.active ? battle.farSide.active[0] : null;
		if (!oc || oc.fainted || !(oc.hp > 0)) return false;
		return true;
	}

	function sendChoice(room, choice) {
		room.choice = {choices: [choice], switchFlags: {}, switchOutFlags: {}, waiting: true};
		room.sendDecision([choice]);
		try { room.closeNotification('choice'); } catch (e) {}
		try { room.updateControlsForPlayer(); } catch (e2) {}
	}

	function hookRoom(room, st) {
		if (room.__psbotOrigReceive) return;
		var orig = room.receive;
		room.__psbotOrigReceive = orig;
		room.receive = function (data) {
			try { onRoomData(room, st, '' + data); } catch (e) {}
			return orig.apply(this, arguments);
		};
	}
	function unhookRoom(room) {
		if (room && room.__psbotOrigReceive) {
			room.receive = room.__psbotOrigReceive;
			delete room.__psbotOrigReceive;
		}
	}

	function myUserid() {
		try { return W.app.user.get('userid'); } catch (e) { return ''; }
	}

	function onRoomData(room, st, data) {
		if (data.indexOf('|error|') >= 0 && /\|error\|\[(Invalid|Unavailable) choice\]/.test(data)) {
			var m = /\|error\|\[(?:Invalid|Unavailable) choice\]([^\n]*)/.exec(data);
			st.errors = (st.errors || 0) + 1;
			st.rejections = (st.rejections || 0) + 1;
			(st.events = st.events || []).push({turn: st.turnNow || 0, type: 'rejected', choice: st.lastChoice || '', reason: m ? m[1].trim() : ''});
			if (st.lastChoice) {
				st.badChoices = st.badChoices || {};
				st.badChoices[st.lastChoice.replace(/ (mega|megax|megay|ultra|terastallize|dynamax|zmove)/g, '')] = true;
			}
			if (m && /trapped|can't switch/i.test(m[1]) && room.request) st.noSwitchRqid = room.request.rqid;
			st.handledReq = null;
			st.readySince = 0;
			log('Server rejected "' + st.lastChoice + '":' + (m ? m[1] : '') + ' - picking again');
		}
		var oppSide = room.side === 'p1' ? 'p2' : room.side === 'p2' ? 'p1' : '';
		if (oppSide && data.indexOf('|request|') !== 0) {
			var lines = data.split('\n');
			for (var i = 0; i < lines.length; i++) {
				var ln = lines[i];
				if (ln.indexOf('|turn|') === 0) {
					st.turnNow = +ln.slice(6) || 0;
					st.oppMoved = false;
					st.oppFaintPending = false;
					st.suckerAt = '';
				} else if (ln.indexOf('|move|' + oppSide) === 0) {
					st.oppMoved = true;
					// Which moves it has used since it came in: one move only fits a Choice item,
					// two rule it out.
					if (ln.indexOf('[from]') < 0) {
						var mp = ln.split('|');
						var who = (mp[2] || '').replace(/^(p\d)[a-z]: /, '$1: ');
						var mid = toID(mp[3]);
						if (!st.oppSince || st.oppSince.ident !== who) st.oppSince = {ident: who, moves: []};
						if (mid && mid !== 'struggle' && st.oppSince.moves.indexOf(mid) < 0) st.oppSince.moves.push(mid);
					}
				} else if (ln.indexOf('|move|' + room.side) === 0) {
					var sp = ln.split('|');
					st.suckerAt = toID(sp[3]) === 'suckerpunch' || toID(sp[3]) === 'thunderclap' ? (st.oppIdent || 'x') : '';
				} else if (ln.indexOf('|-fail|' + room.side) === 0 && st.suckerAt) {
					// Sucker Punch found them not attacking. They will do it again.
					var sk = toID(st.suckerAt.replace(/^p\d: /, ''));
					(st.suckerFails = st.suckerFails || {})[sk] = (st.suckerFails[sk] || 0) + 1;
					st.suckerAt = '';
				} else if (ln.indexOf('|drag|' + oppSide) === 0 || ln.indexOf('|replace|' + oppSide) === 0) {
					st.oppSince = null;
					st.oppIdent = (ln.split('|')[2] || '').replace(/^(p\d)[a-z]: /, '$1: ');
				} else if (ln.indexOf('|faint|' + oppSide) === 0) {
					st.oppFaintPending = true;
				} else if (ln.indexOf('|switch|' + oppSide) === 0) {
					st.oppSince = null;
					st.oppIdent = (ln.split('|')[2] || '').replace(/^(p\d)[a-z]: /, '$1: ');
					if (st.oppFaintPending) st.oppFaintPending = false;
					else if ((st.turnNow || 0) >= 1 && !st.oppMoved) {
						(st.oppSwitchTurns = st.oppSwitchTurns || []).push(st.turnNow);
						(st.oppSwitchIns = st.oppSwitchIns || {})[st.turnNow] = toID((ln.split('|')[3] || '').split(',')[0]);
					}
				}
			}
		}
		// A ladder game ends with "<name>'s rating: 1500 &rarr; <strong>1516</strong>".
		if (data.indexOf("'s rating: ") >= 0) {
			var rre = /\|raw\|([^\n|]*?)'s rating: (\d+(?:\.\d+)?) (?:&rarr;|\u2192) <strong>(\d+(?:\.\d+)?)<\/strong>/g;
			var rm;
			while ((rm = rre.exec(data))) {
				if (toID(rm[1].replace(/&[#a-z0-9]+;/gi, '')) === myUserid()) st.elo = {before: Math.round(+rm[2]), after: Math.round(+rm[3])};
			}
		}
		var w = /\|win\|([^\n]*)/.exec(data);
		if (w && !st.result) {
			st.result = toID(w[1]) === myUserid() ? 'win' : 'loss';
		} else if (/(^|\n)\|tie(\||\n|$)/.test(data) && !st.result) { // not the "|tier|" line every battle starts with
			st.result = 'tie';
		}
	}

	// ------------------------------------------------------------------ battle recorder

	// Every game the bot is in is written down in full: both teams, every decision with the
	// scores behind it, the turn-by-turn history, the raw protocol log, the result and the
	// rating change. Records go to IndexedDB in this browser, and to a file on disk when the
	// page was opened by PSBot.exe (which exposes psbotSave).
	var recorder = {saved: 0, toDisk: 0, error: ''};
	var REC_DB = 'psbot', REC_STORE = 'battles';

	function recDb() {
		return new Promise(function (resolve, reject) {
			if (!W.indexedDB) { reject(new Error('no indexedDB')); return; }
			var open = W.indexedDB.open(REC_DB, 1);
			open.onupgradeneeded = function () { open.result.createObjectStore(REC_STORE, {keyPath: 'id'}); };
			open.onsuccess = function () { resolve(open.result); };
			open.onerror = function () { reject(open.error); };
		});
	}
	function recCount() {
		recDb().then(function (db) {
			var rq = db.transaction(REC_STORE, 'readonly').objectStore(REC_STORE).count();
			rq.onsuccess = function () { recorder.saved = rq.result; db.close(); };
			rq.onerror = function () { db.close(); };
		}).catch(function () {});
	}
	function recAll() {
		return recDb().then(function (db) {
			return new Promise(function (resolve, reject) {
				var rq = db.transaction(REC_STORE, 'readonly').objectStore(REC_STORE).getAll();
				rq.onsuccess = function () { db.close(); resolve(rq.result || []); };
				rq.onerror = function () { db.close(); reject(rq.error); };
			});
		});
	}

	function speciesOf(details) { return ('' + (details || '')).split(',')[0]; }
	function round3(x) { return typeof x === 'number' && isFinite(x) ? Math.round(x * 1000) / 1000 : null; }

	// What both sides look like at the moment of a decision.
	function recSnapshot(room, req) {
		var far = room.battle.farSide;
		return {
			mine: ((req.side && req.side.pokemon) || []).map(function (p) {
				return {species: speciesOf(p.details), condition: p.condition, active: !!p.active};
			}),
			opp: far ? (far.pokemon || []).map(function (q) {
				return {species: q.speciesForme, hp: q.maxhp ? Math.round(100 * q.hp / q.maxhp) : 0, status: q.status || '', active: !!(far.active && far.active[0] === q), fainted: !!q.fainted};
			}) : []
		};
	}

	// The part of the server's request the choice was made from: which moves could be clicked and
	// with how much PP, whether I was trapped, and whether Tera was still available.
	function recRequest(req) {
		var a = req.active && req.active[0];
		if (!a) return {forceSwitch: !!req.forceSwitch};
		return {
			moves: (a.moves || []).map(function (m) { return {id: toID(m.id || m.move), pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled}; }),
			trapped: !!a.trapped, maybeTrapped: !!a.maybeTrapped, canTera: a.canTerastallize || '', canDynamax: !!a.canDynamax
		};
	}

	function recordDecision(st, room, req, d, sent, ms) {
		if (!cfg.record) return;
		var rec = st.rec || (st.rec = {decisions: []});
		try {
			if (req.side && req.side.pokemon) {
				var team = JSON.parse(JSON.stringify(req.side.pokemon));
				if (!rec.team) rec.team = team; // as dealt: moves, items, abilities, stats, Tera types
				rec.teamEnd = team;
			}
			var snap = recSnapshot(room, req);
			rec.decisions.push({
				turn: room.battle.turn, rqid: req.rqid, forced: !!req.forceSwitch, time: new Date().toISOString(),
				choice: d.choice, why: d.why, sent: !!sent, ms: ms === undefined ? null : Math.round(ms),
				predict: d.predict ? {name: d.predict.name, w: round3(d.predict.w)} : null,
				options: (d.table || []).slice(0, 12).map(function (o) {
					return {label: o.label, choice: o.choice, v: round3(o.V), ko: round3(o.pKO), dead: round3(o.pDead), dealt: round3(o.dealt),
						vStay: round3(o.Vstay), vSw: round3(o.Vsw), dmg: o.r ? [Math.round(o.r.min), Math.round(o.r.max)] : null, acc: o.r ? round3(o.r.acc) : null};
				}),
				req: recRequest(req), dbg: d.dbg || null,
				mine: snap.mine, opp: snap.opp
			});
			stashRec(st);
		} catch (e) { recorder.error = '' + (e && e.message || e); }
	}

	// Ratings from the "|player|p1|Name|avatar|1415" lines of a rated game.
	function recRatings(lines, mySide) {
		var out = {mine: null, theirs: null};
		for (var i = 0; i < lines.length && i < 60; i++) {
			var p = ('' + lines[i]).split('|');
			if (p[1] !== 'player' || !p[5]) continue;
			var r = parseInt(p[5], 10);
			if (!isFinite(r)) continue;
			if (p[2] === mySide) out.mine = r; else out.theirs = r;
		}
		return out;
	}

	// The protocol log, boiled down to who did what on each turn.
	function recHistory(lines) {
		var out = [], cur = {turn: 0, events: []};
		var who = function (ident) { return ('' + ident).slice(0, 2) + ' ' + ('' + ident).replace(/^p\d[a-z]?: /, ''); };
		lines.forEach(function (ln) {
			var p = ('' + ln).split('|');
			var k = p[1];
			if (k === 'turn') { out.push(cur); cur = {turn: +p[2] || 0, events: []}; return; }
			if (k === 'move') cur.events.push(who(p[2]) + ' used ' + p[3] + (/\[miss\]/.test(ln) ? ' (missed)' : ''));
			else if (k === 'switch' || k === 'drag') cur.events.push(('' + p[2]).slice(0, 2) + (k === 'drag' ? ' was dragged out to ' : ' sent in ') + speciesOf(p[3]) + ' (' + (p[4] || '') + ')');
			else if (k === 'faint') cur.events.push(who(p[2]) + ' fainted');
			else if (k === '-terastallize') cur.events.push(who(p[2]) + ' Terastallized into ' + p[3]);
			else if (k === '-mega') cur.events.push(who(p[2]) + ' Mega Evolved');
			else if (k === 'cant') cur.events.push(who(p[2]) + ' could not move (' + p[3] + ')');
			else if (k === '-damage' || k === '-heal') cur.events.push('  ' + who(p[2]) + (k === '-damage' ? ' down to ' : ' up to ') + (p[3] || '') + (p[4] ? ' ' + p[4] : ''));
			else if (k === '-status') cur.events.push('  ' + who(p[2]) + ' got ' + p[3]);
			else if (k === '-boost' || k === '-unboost') cur.events.push('  ' + who(p[2]) + ' ' + p[3] + (k === '-boost' ? ' +' : ' -') + p[4]);
			else if (k === '-supereffective' || k === '-resisted' || k === '-immune' || k === '-crit' || k === '-miss') cur.events.push('  ' + k.slice(1));
			else if (k === '-weather' || k === '-fieldstart' || k === '-fieldend' || k === '-sidestart' || k === '-sideend') cur.events.push('  ' + k.slice(1) + ' ' + p.slice(2).join(' '));
			else if (k === '-item' || k === '-enditem' || k === '-ability') cur.events.push('  ' + who(p[2]) + ' ' + k.slice(1) + ' ' + p[3]);
			else if (k === 'win') cur.events.push(p[2] + ' won');
			else if (k === 'tie') cur.events.push('tie');
		});
		out.push(cur);
		return out;
	}

	function buildRecord(id, room, st) {
		var b = room.battle, far = b.farSide, near = b.nearSide;
		var rec = st.rec || {decisions: []};
		var lines = (b.stepQueue || []).slice();
		var ratings = recRatings(lines, room.side || '');
		return {
			v: 2, bot: VERSION, build: BUILD, id: id, format: id.split('-')[1] || '', tier: b.tier || '', gen: b.gen || 0, rated: !!b.rated,
			ratings: ratings, events: st.events || [], rejections: st.rejections || 0, resumed: !!st.resumed,
			arm: st.arm || '', arms: st.arm && cfg.arms ? cfg.arms.id : '',
			server: (W.Config && W.Config.server && W.Config.server.id) || location.host,
			startedAt: new Date(st.startedAt || Date.now()).toISOString(), endedAt: new Date().toISOString(),
			me: {name: near ? near.name : '', side: room.side || ''},
			opponent: {name: far ? far.name : ''},
			result: st.result || 'unknown', elo: st.elo || null, turns: b.turn || 0,
			myTeam: rec.team || null, myTeamEnd: rec.teamEnd || null,
			oppTeam: far ? (far.pokemon || []).map(function (q) {
				return {
					species: q.speciesForme, level: q.level, gender: q.gender || '',
					hp: q.maxhp ? Math.round(100 * q.hp / q.maxhp) : 0, status: q.status || '', fainted: !!q.fainted,
					item: q.item || '', prevItem: q.prevItem || '', prevItemEffect: q.prevItemEffect || '',
					ability: q.ability || '', baseAbility: q.baseAbility || '', tera: q.terastallized || '',
					moves: (q.moveTrack || []).map(function (m) { return {move: ('' + m[0]).replace(/^\*/, ''), used: m[1]}; })
				};
			}) : [],
			oppTeamSize: far ? (far.totalPokemon || 6) : 6,
			decisions: rec.decisions,
			history: recHistory(lines),
			log: lines,
			settings: {parallel: cfg.parallel, fast: cfg.fast, dryRun: cfg.dryRun, switchMargin: cfg.switchMargin, teraMargin: cfg.teraMargin,
				predictSwitch: cfg.predictSwitch, switchPressure: cfg.switchPressure, predictTera: cfg.predictTera}
		};
	}

	function saveRecord(rec) {
		var summary = {
			id: rec.id, endedAt: rec.endedAt, format: rec.format, rated: rec.rated, me: rec.me.name, opponent: rec.opponent.name,
			result: rec.result, turns: rec.turns, eloBefore: rec.elo ? rec.elo.before : null, eloAfter: rec.elo ? rec.elo.after : null,
			bot: rec.bot, build: rec.build, oppRating: rec.ratings ? rec.ratings.theirs : null,
			myTeam: (rec.myTeam || []).map(function (p) { return speciesOf(p.details); }),
			oppTeam: rec.oppTeam.map(function (p) { return p.species; }), decisions: rec.decisions.length
		};
		dropRec(rec.id);
		var onDisk = false;
		try {
			if (typeof W.psbotSave === 'function') {
				W.psbotSave(JSON.stringify({kind: 'battle', id: rec.id, day: rec.endedAt.slice(0, 10), summary: JSON.stringify(summary), body: JSON.stringify(rec, null, 1)}));
				recorder.toDisk++;
				onDisk = true;
			}
		} catch (e) { recorder.error = 'disk: ' + (e && e.message || e); }
		// With the exe writing files, the browser copy would only grow without limit over a long
		// run, so it is skipped. Without the exe, IndexedDB is the only copy.
		if (onDisk) return;
		recDb().then(function (db) {
			var tx = db.transaction(REC_STORE, 'readwrite');
			tx.objectStore(REC_STORE).put(rec);
			tx.oncomplete = function () { db.close(); recCount(); };
			tx.onerror = function () { recorder.error = 'db: ' + tx.error; db.close(); };
		}).catch(function (e) { recorder.error = 'db: ' + (e && e.message || e); });
	}

	// Hands every recorded battle over as one JSON file.
	function exportRecords() {
		return recAll().then(function (all) {
			var a = document.createElement('a');
			a.href = URL.createObjectURL(new Blob([JSON.stringify(all)], {type: 'application/json'}));
			a.download = 'psbot-battles-' + new Date().toISOString().slice(0, 10) + '.json';
			document.body.appendChild(a);
			a.click();
			a.parentNode.removeChild(a);
			setTimeout(function () { URL.revokeObjectURL(a.href); }, 5000);
			log('Exported ' + all.length + ' recorded battles');
			return all.length;
		});
	}

	// ------------------------------------------------------------------ rating history

	function eloStoreKey() {
		return 'psbot-elo-' + (myUserid() || 'guest');
	}
	function loadElo() {
		var who = myUserid();
		if (!who || elo.loadedFor === who) return;
		elo.loadedFor = who;
		try {
			var d = JSON.parse(W.localStorage.getItem(eloStoreKey()) || '[]');
			if (Array.isArray(d)) elo.history = d.slice(-500);
		} catch (e) {}
	}
	function recordElo(st, fmt, foe) {
		loadElo();
		var e = {t: Date.now(), fmt: fmt, before: st.elo.before, after: st.elo.after, result: st.result || '', foe: foe || ''};
		elo.history.push(e);
		if (elo.history.length > 500) elo.history.shift();
		if (elo.sessionStart === null) elo.sessionStart = e.before;
		try { W.localStorage.setItem(eloStoreKey(), JSON.stringify(elo.history)); } catch (e2) {}
		var d = e.after - e.before;
		log('Elo ' + e.before + ' -> ' + e.after + ' (' + (d >= 0 ? '+' : '') + d + ')');
	}
	// The games to draw: the most recent run of entries in the same format.
	function eloSeries() {
		loadElo();
		var h = elo.history;
		if (!h.length) return null;
		var fmt = h[h.length - 1].fmt;
		var games = h.filter(function (e) { return e.fmt === fmt; }).slice(-100);
		return {fmt: fmt, games: games, points: [games[0].before].concat(games.map(function (e) { return e.after; }))};
	}
	function eloChart(series, w, h) {
		var pts = series.points;
		if (pts.length < 2) return '';
		var min = Math.min.apply(null, pts), max = Math.max.apply(null, pts);
		if (max - min < 20) { var mid = (max + min) / 2; min = mid - 10; max = mid + 10; }
		var padL = 34, padR = 8, padT = 9, padB = 9;
		var x = function (i) { return padL + (w - padL - padR) * i / (pts.length - 1); };
		var y = function (v) { return padT + (h - padT - padB) * (1 - (v - min) / (max - min)); };
		var path = pts.map(function (v, i) { return (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(v).toFixed(1); }).join(' ');
		var dots = series.games.map(function (g, i) {
			var c = g.result === 'win' ? '#68d391' : g.result === 'loss' ? '#fc8181' : '#cbd5e0';
			return '<circle cx="' + x(i + 1).toFixed(1) + '" cy="' + y(g.after).toFixed(1) + '" r="2.4" fill="' + c + '"><title>' +
				escHtml((g.result || 'game') + ' vs ' + (g.foe || '?') + ': ' + g.before + ' to ' + g.after) + '</title></circle>';
		}).join('');
		var label = function (v, yy) { return '<text x="3" y="' + yy + '" fill="#a0aec0" font-size="9" font-family="Verdana,sans-serif">' + Math.round(v) + '</text>'; };
		// The dashed line marks where this run of games started.
		return '<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%" style="display:block;background:#0f141b;border-radius:6px;margin-top:5px">' +
			'<line x1="' + padL + '" y1="' + y(pts[0]).toFixed(1) + '" x2="' + (w - padR) + '" y2="' + y(pts[0]).toFixed(1) + '" stroke="#4a5568" stroke-dasharray="3 3"/>' +
			label(max, padT + 3) + label(min, h - padB + 3) +
			'<path d="' + path + '" fill="none" stroke="#63b3ed" stroke-width="1.6" stroke-linejoin="round"/>' + dots + '</svg>';
	}
	function eloPanelHtml(sr) {
		if (!sr) return '';
		var g = sr.games, last = g[g.length - 1];
		var d = last.after - last.before;
		var sess = elo.sessionStart === null ? null : last.after - elo.sessionStart;
		var peak = Math.max.apply(null, sr.points);
		var sign = function (n) { return (n >= 0 ? '+' : '') + n; };
		var tone = function (n) { return n >= 0 ? 'pb-up' : 'pb-down'; };
		return '<div><b>' + last.after + '</b> <span class="' + tone(d) + '">' + sign(d) + '</span>' +
			(sess === null ? '' : ' &middot; this session <span class="' + tone(sess) + '">' + sign(sess) + '</span>') +
			' &middot; peak ' + peak + ' &middot; ' + g.length + ' game' + (g.length === 1 ? '' : 's') + '</div>' +
			eloChart(sr, 328, g.length >= 8 ? 104 : 76);
	}
	function isMyBattle(battle) {
		var me = myUserid();
		if (!me) return false;
		return (battle.p1 && toID(battle.p1.name) === me) || (battle.p2 && toID(battle.p2.name) === me);
	}

	// Ask for the battle timer once per game, so an opponent who stops moving cannot hold the
	// bot in one battle forever. It is a room command, not a chat message.
	function maybeStartTimer(room, st, battle) {
		if (battle.kickingInactive) {
			if (!st.timerOn) { st.timerOn = true; log('Timer is on'); }
			return;
		}
		st.timerOn = false;
		if (!(battle.turn >= 1) || (st.timerTries || 0) >= 2) return;
		if (st.timerAt && Date.now() - st.timerAt < 20000) return;
		st.timerAt = Date.now();
		st.timerTries = (st.timerTries || 0) + 1;
		try { room.send('/timer on'); } catch (e) {
			try { W.app.send('/timer on', room.id); } catch (e2) {}
		}
	}

	// Bookkeeping after a choice is made. The page loop and the headless engine both use it, so
	// the switch gates in decide() see the same history either way.
	function noteChoice(st, req, choice, turn) {
		st.lastChoice = choice;
		if (/^switch/.test(choice) && !req.forceSwitch) {
			st.switchStreak = (st.switchStreak || 0) + 1;
			var leaving = ((req.side && req.side.pokemon) || []).filter(function (q) { return q.active; })[0];
			if (leaving) (st.leftAt = st.leftAt || {})[leaving.ident] = turn;
		} else if (/^move/.test(choice)) {
			st.switchStreak = 0;
		}
	}

	// Catch the battle's bookkeeping up with everything already in its log. Needed for a battle
	// the bot joins part-way: after a page reload, or when the first lines arrived before the
	// client knew which side is mine.
	function syncRoom(room, st) {
		st.oppSwitchTurns = [];
		st.oppSwitchIns = {};
		st.suckerFails = {};
		st.oppSince = null;
		st.oppIdent = '';
		var lines = (room.battle.stepQueue || []).filter(function (l) { return ('' + l).indexOf('|request|') !== 0 && ('' + l).indexOf('|error|') !== 0; });
		if (lines.length) { try { onRoomData(room, st, lines.join('\n')); } catch (e) {} }
	}

	// Things that would otherwise stop an unattended run: a dropped connection, a search that
	// never matches, a page that has been open for hours.
	function watchdog(now) {
		if (!cfg.watchdog || run.reloading) return;
		var down = false;
		try { down = !!W.app.isDisconnected; } catch (e) {}
		var running = cfg.autoSearch || ui.live > 0 || run.resumed;
		if (down) {
			if (!run.downSince) { run.downSince = now; event('disconnected', {live: ui.live, auto: cfg.autoSearch}); }
			// Reload to reconnect, backing off when it keeps happening.
			var recent = run.reloadTimes.filter(function (t) { return now - t < 30 * 60000; }).length;
			var wait = [15, 30, 60, 120, 300, 600][Math.min(5, recent)] * 1000;
			if (running && now - run.downSince > wait) reloadPage('disconnected for ' + Math.round((now - run.downSince) / 1000) + 's');
			return;
		}
		run.downSince = 0;
		if (!cfg.enabled) return;
		// A search that has been open for six minutes is not going to match: start it again.
		if (ui.searching) {
			ui.searchingSince = ui.searchingSince || now;
			if (now - ui.searchingSince > 6 * 60000) {
				ui.searchingSince = 0;
				event('search-stuck', 'cancelling a search that found nothing in six minutes');
				try { W.app.send('/cancelsearch'); } catch (e2) {}
				lastSearchAt = now;
			}
		} else {
			ui.searchingSince = 0;
		}
		// A fresh page every few hours, taken between games.
		if (cfg.autoSearch && cfg.reloadHours > 0 && now - run.loadedAt > cfg.reloadHours * 3600000) {
			if (!ui.draining) { ui.draining = true; event('drain', 'finishing the running games before a scheduled page reload'); }
		}
		if (ui.draining && ui.searching && !ui.drainCancelled) {
			ui.drainCancelled = true;
			try { W.app.send('/cancelsearch'); } catch (e3) {}
		}
		if (ui.draining && ui.live === 0 && !ui.searching && now - lastSearchAt > 20000) reloadPage('scheduled refresh after ' + Math.round((now - run.loadedAt) / 3600000 * 10) / 10 + 'h');
		// Forget battles whose rooms are gone.
		if (now - run.lastPrune > 60000) {
			run.lastPrune = now;
			Object.keys(rooms).forEach(function (rid) {
				if (W.app.rooms[rid]) return;
				var s2 = rooms[rid];
				if (s2.finished || now - (s2.lastSeen || s2.startedAt || 0) > 10 * 60000) delete rooms[rid];
			});
		}
	}

	function tick() {
		var nowT = Date.now();
		run.lastTickAt = nowT;
		if (nowT - run.lastBeat > 30000) {
			run.lastBeat = nowT;
			saveRun();
			toDisk('status', {body: JSON.stringify(statusNow())});
		}
		if (nowT - run.lastLogFlush > 10000) { run.lastLogFlush = nowT; flushLog(); }
		if (!W.app || !W.app.rooms) return;
		try { watchdog(nowT); } catch (e) { run.errors++; }
		if (!cfg.enabled || run.reloading) return;
		hookPopups();
		var live = 0;
		var starting = 0;
		var liveNow = Object.keys(W.app.rooms).some(function (rid) {
			var rm = W.app.rooms[rid];
			return rid.indexOf('battle-') === 0 && rm && rm.battle && rm.side && !(rm.battleEnded || rm.battle.ended);
		});
		Object.keys(W.app.rooms).forEach(function (id) {
			var room = W.app.rooms[id];
			if (!room || !room.battle || id.indexOf('battle-') !== 0) return;
			var st = rooms[id];
			if (!st) {
				st = rooms[id] = {id: id, startedAt: Date.now()};
				// A game that was already under way when the page loaded: take its record back.
				var kept = unstashRec(id);
				if (kept) { st.rec = kept.rec; st.startedAt = kept.startedAt || st.startedAt; st.played = true; st.resumed = true; }
			}
			st.lastSeen = nowT;
			if (st.arm === undefined || st.armsId !== (cfg.arms ? cfg.arms.id : '')) { st.arm = armOf(id); st.armsId = cfg.arms ? cfg.arms.id : ''; }
			hookRoom(room, st);
			if (!st.synced && room.side) { st.synced = true; syncRoom(room, st); }
			var battle = room.battle;
			var roomFmt = id.split('-')[1] || toID(battle.tier);
			loadSets(roomFmt, battle.gen || 9);

			if (room.battleEnded || battle.ended) {
				if (!st.finished && (st.played || st.result)) {
					st.finished = true;
					st.finishedAt = Date.now();
					record.games++;
					if (st.result === 'win') record.wins++; else if (st.result === 'loss') record.losses++; else record.ties++;
					var foe = battle.farSide ? battle.farSide.name : '';
					log('Game over (' + (st.result || 'ended') + ') vs ' + foe + '. Record ' + record.wins + 'W ' + record.losses + 'L' + (record.ties ? ' ' + record.ties + 'T' : ''));
					saveRun();
				}
				if (st.elo && !st.eloLogged) {
					st.eloLogged = true;
					recordElo(st, id.split('-')[1] || '', battle.farSide ? battle.farSide.name : '');
				}
				if (cfg.record && st.finished && !st.saved && (st.elo || !battle.rated || Date.now() - st.finishedAt > 8000)) {
					st.saved = true;
					try {
						saveRecord(buildRecord(id, room, st));
						log('Recorded ' + id + ' (' + (st.result || 'ended') + ', ' + ((st.rec && st.rec.decisions.length) || 0) + ' decisions)');
					} catch (e) {
						recorder.error = '' + (e && e.message || e);
						log('Could not record ' + id + ': ' + recorder.error);
					}
				}
				// Close this tab once a newer game is running or the bot is about to queue one.
				// The last game stays on screen otherwise. The delay leaves time for the rating
				// line to arrive.
				if (cfg.closeFinished && !st.closed && (liveNow || cfg.autoSearch) && (st.finished || isMyBattle(battle)) && (st.saved || !cfg.record || !st.finished)) {
					st.endSeenAt = st.endSeenAt || Date.now();
					// A rated game's rating line comes from the login server and can lag a little.
					var hold = battle.rated && !st.elo ? 15000 : 6000;
					if (Date.now() - st.endSeenAt > hold) {
						st.closed = true;
						try { W.app.leaveRoom(id); } catch (e) {}
					}
				}
				return;
			}
			if (!room.side) {
				// Either a battle being watched, or one of ours whose first request is not in yet.
				if (Date.now() - st.startedAt < 8000) starting++;
				return;
			}
			live++;
			if (cfg.autoTimer && !cfg.dryRun) maybeStartTimer(room, st, battle);
			var req = room.request;
			if (!req || req.requestType === 'wait' || req.wait) return;
			if (st.lastRqid !== req.rqid) {
				st.lastRqid = req.rqid;
				st.errors = 0;
				st.badChoices = {};
				st.readySince = 0;
				st.reqAt = nowT;
				st.rescued = false;
			}
			// Last resort: a request that has sat unanswered for a minute and a half gets the
			// server's default choice, so a bug here can never time a game out.
			if (cfg.watchdog && !cfg.dryRun && st.reqAt && nowT - st.reqAt > 90000 && !st.rescued && !(room.choice && room.choice.waiting)) {
				st.rescued = true;
				st.pending = null;
				(st.events = st.events || []).push({turn: battle.turn, type: 'rescued', after: nowT - st.reqAt});
				event('rescue', {room: id, turn: battle.turn, waited: nowT - st.reqAt});
				try { sendChoice(room, 'default'); st.handledReq = req; st.played = true; } catch (e) {}
				return;
			}
			if (room.choice && room.choice.waiting) { st.handledReq = req; st.pending = null; return; } // someone already chose
			if (st.pending) {
				if (st.pending.req !== req) { st.pending = null; } else {
					if (Date.now() < st.pending.at) return;
					var pc = st.pending.choice;
					st.pending = null;
					try { sendChoice(room, pc); } catch (e) { console.error('[PSBot] send failed', e); st.handledReq = null; }
					return;
				}
			}
			// Dry run switched off mid-turn: decide this request again and actually send it.
			if (!cfg.dryRun && st.dryReq === req) { st.dryReq = null; st.handledReq = null; }
			if (req === st.handledReq) return;

			if (!isSynced(room, req)) {
				st.readySince = 0;
				st.waitingSince = st.waitingSince || Date.now();
				// Background tabs barely animate, and the client pauses every battle that is not
				// the one on screen, so neither would reach the end of the turn. Jump straight to
				// the current state instead.
				if (!battle.atQueueEnd && battle.seeking === null &&
					(cfg.fast || document.hidden || W.app.curRoom !== room || Date.now() - st.waitingSince > cfg.maxAnimWaitMs)) {
					try { battle.seekTurn(Infinity); } catch (e) {}
				}
				// If the state never lines up, decide anyway rather than time out.
				if (Date.now() - st.waitingSince < 20000 || !battle.atQueueEnd) return;
			}
			// Opposing movesets come from the set data, so give the download a moment on turn one.
			if (setsLoading[roomFmt] && Date.now() - setsLoading[roomFmt] < 5000) return;
			if (movesLoading && Date.now() - movesLoading < 5000) return;
			if (!st.readySince) { st.readySince = Date.now(); return; }
			if (Date.now() - st.readySince < cfg.settleMs) return;
			st.waitingSince = 0;

			var d, t0 = W.performance && W.performance.now ? W.performance.now() : Date.now();
			try {
				d = (st.errors || 0) >= 3 ? {choice: 'default', why: 'fallback after repeated rejections', table: []} : decide(room, req, st);
			} catch (e) {
				d = {choice: 'default', why: 'error: ' + (e && e.message), table: []};
				run.errors++;
				(st.events = st.events || []).push({turn: battle.turn, type: 'error', message: '' + (e && e.message)});
				event('decide-error', {room: id, turn: battle.turn, message: '' + (e && e.message), stack: ('' + (e && e.stack || '')).split('\n').slice(0, 6).join(' | ')});
				console.error('[PSBot] decide failed', e);
			}
			var took = (W.performance && W.performance.now ? W.performance.now() : Date.now()) - t0;
			run.decisions++;
			run.lastDecisionAt = Date.now();
			if (took > run.slowest) run.slowest = Math.round(took);
			lastExplain = {room: id, turn: battle.turn, decision: d};
			st.lastWhy = d.why;
			var top = (d.table || []).slice(0, 4).map(function (o) { return o.label + ' ' + o.V.toFixed(2); }).join(' | ');
			log('T' + battle.turn + ' ' + (cfg.dryRun ? '[dry run] ' : '') + '-> ' + d.choice + '  (' + d.why + ')' + (top ? '  [' + top + ']' : ''));
			st.handledReq = req;
			noteChoice(st, req, d.choice, battle.turn);
			recordDecision(st, room, req, d, !cfg.dryRun, took);
			if (cfg.dryRun) { st.dryReq = req; return; }
			st.played = true;
			st.pending = {req: req, choice: d.choice, at: Date.now() + cfg.thinkMs};
		});

		// ---------- queue games, keeping up to cfg.parallel running at once
		var menu = W.app.rooms[''];
		var searching = !!(menu && menu.searching && (!Array.isArray(menu.searching) || menu.searching.length));
		ui.live = live;
		ui.searching = searching;
		if (cfg.autoSearch && !cfg.dryRun) {
			var want = clampInt(cfg.parallel, 1, MAX_PARALLEL);
			if (ui.serverCap) want = Math.min(want, ui.serverCap);
			var done = cfg.maxGames > 0 && record.searches - queueBase >= cfg.maxGames;
			if (done) {
				if (live === 0 && starting === 0 && !searching) {
					cfg.autoSearch = false;
					log('All ' + cfg.maxGames + ' games played, auto-queue off.');
					var srDone = eloSeries();
					if (srDone && elo.sessionStart !== null) {
						var endElo = srDone.games[srDone.games.length - 1].after;
						log('Run finished: ' + record.wins + 'W ' + record.losses + 'L, Elo ' + elo.sessionStart + ' -> ' + endElo + ' (' + (endElo - elo.sessionStart >= 0 ? '+' : '') + (endElo - elo.sessionStart) + ')');
					}
				}
			} else if (live + starting < want && !searching && !ui.draining && Date.now() - lastSearchAt > searchGap() && Date.now() > ui.blockedUntil &&
				W.app.user && W.app.user.get('named')) {
				var fmt = toID(cfg.format);
				var info = W.BattleFormats && W.BattleFormats[fmt];
				if (info && !info.team) {
					cfg.autoSearch = false;
					log('Format ' + fmt + ' needs your own team; auto-queue only handles random formats.');
				} else {
					lastSearchAt = Date.now();
					lastSearchLive = live + starting;
					record.searches++;
					log('Searching for a ' + fmt + ' game (' + (record.searches - queueBase) + (cfg.maxGames ? '/' + cfg.maxGames : '') + ')' +
						(want > 1 ? ', ' + (live + starting) + ' of ' + want + ' running' : ''));
					W.app.sendTeam(null, function () { W.app.send('/search ' + fmt); });
				}
			}
		}
		renderPanel();
	}

	// The public server refuses more than 12 searches per 3 minutes, so space them out there.
	function searchGap() {
		return W.Config && W.Config.server && W.Config.server.id === 'showdown' ? 16000 : 2500;
	}

	// ------------------------------------------------------------------ on-page panel

	var SETTINGS_KEY = 'psbot-settings';
	var SAVED = ['format', 'parallel', 'maxGames', 'autoTimer', 'closeFinished', 'fast'];
	var ui = {el: null, refs: {}, cache: {}, collapsed: false, pos: null, live: 0, searching: false, serverCap: 0, blockedUntil: 0, logVer: 0, shownLogVer: -1, formatsKey: ''};

	function loadSettings() {
		try {
			var d = JSON.parse(W.localStorage.getItem(SETTINGS_KEY) || '{}');
			SAVED.forEach(function (k) { if (d[k] !== undefined && typeof d[k] === typeof cfg[k]) cfg[k] = d[k]; });
			ui.collapsed = !!d.collapsed;
			if (d.pos && typeof d.pos.x === 'number' && typeof d.pos.y === 'number') ui.pos = d.pos;
		} catch (e) {}
		cfg.parallel = clampInt(cfg.parallel, 1, MAX_PARALLEL);
		cfg.maxGames = clampInt(cfg.maxGames, 0, 100000);
	}
	function saveSettings() {
		try {
			var d = {collapsed: ui.collapsed, pos: ui.pos};
			SAVED.forEach(function (k) { d[k] = cfg[k]; });
			W.localStorage.setItem(SETTINGS_KEY, JSON.stringify(d));
		} catch (e) {}
	}

	var PANEL_CSS = [
		'#psbot-panel{position:fixed;right:10px;bottom:10px;z-index:99999;width:348px;max-width:calc(100vw - 16px);font:12px/1.4 Verdana,Helvetica,Arial,sans-serif;color:#e2e8f0;background:#171c24;border:1px solid #2d3748;border-radius:10px;box-shadow:0 8px 28px rgba(0,0,0,.55);overflow:hidden;text-align:left}',
		'#psbot-panel *{box-sizing:border-box}',
		'#psbot-panel .pb-head{display:flex;align-items:center;gap:7px;padding:7px 10px;background:#1f2733;cursor:move;user-select:none}',
		'#psbot-panel .pb-dot{width:9px;height:9px;border-radius:50%;background:#718096;flex:none}',
		'#psbot-panel .pb-dot.on{background:#48bb78;box-shadow:0 0 6px #48bb78}',
		'#psbot-panel .pb-title{font-weight:bold;font-size:13px}',
		'#psbot-panel .pb-ver{color:#718096;font-size:10px}',
		'#psbot-panel .pb-rec{margin-left:auto;color:#cbd5e0;font-size:11px;white-space:nowrap}',
		'#psbot-panel .pb-x{cursor:pointer;color:#a0aec0;padding:0 2px 0 6px;font-size:11px}',
		'#psbot-panel .pb-x:hover{color:#fff}',
		'#psbot-panel .pb-body{padding:9px 10px 10px;display:flex;flex-direction:column;gap:8px;max-height:calc(100vh - 64px);overflow-y:auto}',
		'#psbot-panel.pb-min .pb-body{display:none}',
		'#psbot-panel .pb-row{display:flex;align-items:center;gap:6px}',
		'#psbot-panel .pb-btn{font:bold 12px Verdana,Helvetica,Arial,sans-serif;padding:7px 8px;border-radius:7px;border:1px solid #4a5568;background:#2d3748;color:#fff;cursor:pointer;flex:1;text-shadow:none;box-shadow:none}',
		'#psbot-panel .pb-btn:hover{filter:brightness(1.18)}',
		'#psbot-panel .pb-btn.go{background:#2b6cb0;border-color:#3182ce}',
		'#psbot-panel .pb-btn.on{background:#2f855a;border-color:#38a169}',
		'#psbot-panel .pb-btn.stop{background:#9b2c2c;border-color:#c53030}',
		'#psbot-panel .pb-status{color:#a0aec0;font-size:11px;min-height:15px}',
		'#psbot-panel .pb-grid{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center}',
		'#psbot-panel .pb-grid label{color:#a0aec0;font-size:11px;white-space:nowrap}',
		'#psbot-panel select,#psbot-panel input[type=number]{font:12px Verdana,Helvetica,Arial,sans-serif;background:#0f141b;color:#e2e8f0;border:1px solid #4a5568;border-radius:6px;padding:4px 6px;height:26px;min-width:0}',
		'#psbot-panel select{width:100%}',
		'#psbot-panel input[type=number]{width:64px;text-align:center}',
		'#psbot-panel .pb-step{display:inline-flex;align-items:center;gap:4px}',
		'#psbot-panel .pb-step button{width:26px;height:26px;border:1px solid #4a5568;border-radius:6px;background:#2d3748;color:#fff;cursor:pointer;font:bold 14px/1 Verdana,sans-serif;padding:0}',
		'#psbot-panel .pb-step button:hover{filter:brightness(1.2)}',
		'#psbot-panel .pb-hint{color:#718096;font-size:10px;margin-left:6px}',
		'#psbot-panel .pb-checks{display:flex;flex-wrap:wrap;gap:4px 12px}',
		'#psbot-panel .pb-chk{display:inline-flex;align-items:center;gap:4px;color:#cbd5e0;font-size:11px;cursor:pointer;white-space:nowrap}',
		'#psbot-panel .pb-chk input{margin:0}',
		'#psbot-panel .pb-sec{border-top:1px solid #2d3748;padding-top:7px}',
		'#psbot-panel .pb-h{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:#718096;margin-bottom:4px;display:flex;justify-content:space-between}',
		'#psbot-panel .pb-h a{cursor:pointer;color:#718096;text-transform:none;letter-spacing:0}',
		'#psbot-panel .pb-h a:hover{color:#cbd5e0}',
		'#psbot-panel .pb-dim{color:#718096;font-size:11px}',
		'#psbot-panel .pb-up{color:#68d391}',
		'#psbot-panel .pb-down{color:#fc8181}',
		'#psbot-panel .pb-elo b{font-size:15px;color:#fff}',
		'#psbot-panel .pb-live{max-height:132px;overflow-y:auto}',
		'#psbot-panel .pb-game{display:flex;gap:8px;padding:3px 6px;border-radius:5px;cursor:pointer;font-size:11px;align-items:baseline}',
		'#psbot-panel .pb-game:hover{background:#243042}',
		'#psbot-panel .pb-game .pb-foe{flex:0 1 112px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#e2e8f0}',
		'#psbot-panel .pb-game .pb-num{flex:none;color:#a0aec0;font-variant-numeric:tabular-nums}',
		'#psbot-panel .pb-game .pb-last{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#718096;text-align:right}',
		'#psbot-panel .pb-log{max-height:104px;overflow-y:auto;font:10.5px/1.45 Consolas,Menlo,monospace;color:#a0aec0;background:#0f141b;border-radius:6px;padding:6px;white-space:pre-wrap;word-break:break-word}'
	].join('\n');

	var PANEL_HTML =
		'<div class="pb-head" data-ref="head" title="Drag to move">' +
			'<span class="pb-dot" data-ref="dot"></span><span class="pb-title">PSBot</span><span class="pb-ver">' + VERSION + '</span>' +
			'<span class="pb-rec" data-ref="rec"></span><span class="pb-x" data-act="collapse" data-ref="collapse" title="Collapse or expand"></span>' +
		'</div>' +
		'<div class="pb-body">' +
			'<div class="pb-row"><button class="pb-btn" data-act="toggle" data-ref="toggle"></button><button class="pb-btn" data-act="queue" data-ref="queue"></button></div>' +
			'<div class="pb-status" data-ref="status"></div>' +
			'<div class="pb-grid">' +
				'<label>Format</label><select data-set="format" data-ref="format"></select>' +
				'<label>Games at once</label><span class="pb-step"><button data-act="par-" title="Fewer">&minus;</button><input type="number" min="1" max="' + MAX_PARALLEL + '" data-set="parallel" data-ref="parallel"><button data-act="par+" title="More">+</button><span class="pb-hint">max ' + MAX_PARALLEL + '</span></span>' +
				'<label>Games to play</label><span class="pb-step"><input type="number" min="0" data-set="maxGames" data-ref="maxGames"><span class="pb-hint">0 = keep going</span></span>' +
			'</div>' +
			'<div class="pb-checks">' +
				'<label class="pb-chk" title="Turn the battle timer on at the start of every game"><input type="checkbox" data-set="autoTimer" data-ref="autoTimer">Timer</label>' +
				'<label class="pb-chk" title="Close a finished battle tab once the next game is running"><input type="checkbox" data-set="closeFinished" data-ref="closeFinished">Close finished tabs</label>' +
				'<label class="pb-chk" title="Do not wait for battle animations"><input type="checkbox" data-set="fast" data-ref="fast">Skip animations</label>' +
				'<label class="pb-chk" title="Log the picks without sending them"><input type="checkbox" data-set="dryRun" data-ref="dryRun">Dry run</label>' +
			'</div>' +
			'<div class="pb-sec"><div class="pb-h"><span>Rating</span><a data-act="eloreset" title="Forget the saved rating history">reset</a></div><div class="pb-elo" data-ref="elo"></div></div>' +
			'<div class="pb-sec"><div class="pb-h"><span data-ref="liveH"></span></div><div class="pb-live" data-ref="live"></div></div>' +
			'<div class="pb-sec"><div class="pb-h"><span>Recorded games</span><a data-act="recexport" title="Download every recorded game as one JSON file">export</a></div><div class="pb-dim" data-ref="recInfo"></div></div>' +
			'<div class="pb-sec"><div class="pb-h"><span>Log</span><a data-act="logclear">clear</a></div><div class="pb-log" data-ref="log"></div></div>' +
		'</div>';

	function escHtml(s) {
		return ('' + s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
	}

	function buildPanel() {
		var css = document.getElementById('psbot-style');
		if (!css) {
			css = document.createElement('style');
			css.id = 'psbot-style';
			(document.head || document.body).appendChild(css);
		}
		css.textContent = PANEL_CSS;
		var el = document.createElement('div');
		el.id = 'psbot-panel';
		el.innerHTML = PANEL_HTML;
		document.body.appendChild(el);
		ui.el = el;
		ui.refs = {};
		ui.cache = {};
		ui.shownLogVer = -1;
		ui.formatsKey = '';
		Array.prototype.forEach.call(el.querySelectorAll('[data-ref]'), function (n) { ui.refs[n.getAttribute('data-ref')] = n; });
		el.addEventListener('click', onPanelClick);
		el.addEventListener('change', onPanelChange);
		ui.refs.head.addEventListener('mousedown', startDrag);
		placePanel();
	}

	function setAutoQueue(on) {
		cfg.autoSearch = !!on;
		ui.draining = false;
		if (on) {
			queueBase = record.searches;
			lastSearchAt = 0;
			ui.blockedUntil = 0;
			if (!cfg.enabled) api.start();
		}
		run.resumed = false;
		saveRun();
		event(on ? 'queue-on' : 'queue-off', {format: cfg.format, parallel: cfg.parallel});
	}

	function onPanelClick(ev) {
		var n = ev.target;
		while (n && n !== ui.el && !(n.getAttribute && n.getAttribute('data-act'))) n = n.parentNode;
		if (!n || n === ui.el) return;
		var a = n.getAttribute('data-act');
		if (a === 'toggle') { if (cfg.enabled) api.stop(); else api.start(); }
		if (a === 'queue') setAutoQueue(!cfg.autoSearch);
		if (a === 'collapse') { ui.collapsed = !ui.collapsed; saveSettings(); }
		if (a === 'par-' || a === 'par+') { cfg.parallel = clampInt(cfg.parallel + (a === 'par+' ? 1 : -1), 1, MAX_PARALLEL); saveSettings(); }
		if (a === 'eloreset') { elo.history = []; elo.sessionStart = null; try { W.localStorage.removeItem(eloStoreKey()); } catch (e) {} }
		if (a === 'logclear') { logBuf.length = 0; ui.logVer++; }
		if (a === 'recexport') { exportRecords().catch(function (e) { log('Export failed: ' + (e && e.message || e)); }); }
		if (a === 'room') { try { W.app.focusRoom(n.getAttribute('data-room')); } catch (e2) {} }
		renderPanel();
	}

	function onPanelChange(ev) {
		var n = ev.target;
		var k = n && n.getAttribute && n.getAttribute('data-set');
		if (!k) return;
		if (n.type === 'checkbox') cfg[k] = !!n.checked;
		else if (k === 'parallel') cfg.parallel = clampInt(n.value, 1, MAX_PARALLEL);
		else if (k === 'maxGames') cfg.maxGames = clampInt(n.value, 0, 100000);
		else if (k === 'format') {
			cfg.format = toID(n.value) || 'gen9randombattle';
			var g = /^gen(\d+)/.exec(cfg.format);
			loadSets(cfg.format, g ? +g[1] : 9);
		}
		saveSettings();
		if (n.blur && n.type === 'number') n.blur();
		renderPanel();
	}

	function startDrag(ev) {
		if (ev.button !== 0 || (ev.target.getAttribute && ev.target.getAttribute('data-act'))) return;
		var r = ui.el.getBoundingClientRect();
		var dx = ev.clientX - r.left, dy = ev.clientY - r.top;
		var move = function (e) { ui.pos = {x: e.clientX - dx, y: e.clientY - dy}; placePanel(); };
		var up = function () {
			document.removeEventListener('mousemove', move);
			document.removeEventListener('mouseup', up);
			saveSettings();
		};
		document.addEventListener('mousemove', move);
		document.addEventListener('mouseup', up);
		ev.preventDefault();
	}

	// Keeps a dragged panel inside the window, so the header can always be reached.
	function placePanel() {
		var s = ui.el && ui.el.style;
		if (!s) return;
		if (!ui.pos) { s.left = ''; s.top = ''; s.right = ''; s.bottom = ''; return; }
		var x = Math.max(0, Math.min(Math.max(0, W.innerWidth - ui.el.offsetWidth), ui.pos.x));
		var y = Math.max(0, Math.min(Math.max(0, W.innerHeight - 34), ui.pos.y));
		s.left = x + 'px';
		s.top = y + 'px';
		s.right = 'auto';
		s.bottom = 'auto';
	}

	// Random-team formats the ladder offers, for the format dropdown.
	function randomFormats() {
		var out = [];
		var F = W.BattleFormats || {};
		Object.keys(F).forEach(function (id) {
			var f = F[id];
			// The bot plays singles, so leave out the formats with more than one Pokemon a side.
			if (f && f.team && f.searchShow && !/doubles|triples|freeforall|multi|2v2/.test(id)) out.push({id: id, name: f.name || id});
		});
		var cur = toID(cfg.format);
		if (!out.some(function (f) { return f.id === cur; })) out.unshift({id: cur, name: cur});
		return out;
	}

	function statusText() {
		if (!cfg.enabled) return 'Off. Press Start bot and it plays the battles you are in.';
		if (cfg.dryRun) return 'Dry run: picks are logged, nothing is sent.';
		var named = false;
		try { named = !!W.app.user.get('named'); } catch (e) {}
		if (!named) return 'Log in to Showdown first (Choose name, top right).';
		var n = ui.live;
		if (cfg.autoSearch) {
			var want = clampInt(cfg.parallel, 1, MAX_PARALLEL);
			var t = 'Playing ' + n + ' of ' + (ui.serverCap ? Math.min(want, ui.serverCap) : want) + ' at once';
			if (cfg.maxGames) t += ' &middot; ' + Math.min(cfg.maxGames, record.searches - queueBase) + '/' + cfg.maxGames + ' started';
			if (ui.searching) t += ' &middot; searching';
			if (ui.serverCap && want > ui.serverCap) t += ' &middot; server limit ' + ui.serverCap;
			else if (Date.now() < ui.blockedUntil) t += ' &middot; waiting on the server search limit';
			return t;
		}
		return n ? 'Playing ' + n + ' game' + (n === 1 ? '' : 's') + '.' : 'On. Click Battle! or turn Auto-queue on.';
	}

	function liveRows() {
		var rows = [];
		if (!W.app || !W.app.rooms) return rows;
		Object.keys(W.app.rooms).forEach(function (id) {
			var room = W.app.rooms[id];
			if (id.indexOf('battle-') !== 0 || !room || !room.battle || !room.side) return;
			var b = room.battle;
			if (room.battleEnded || b.ended) return;
			var st = rooms[id] || {};
			var mine = 0, total = 0;
			var mons = room.request && room.request.side && room.request.side.pokemon;
			if (mons) { total = mons.length; mons.forEach(function (p) { if (!parseCond(p.condition).fainted) mine++; }); }
			var far = b.farSide;
			var foeTotal = far ? (far.totalPokemon || 6) : 6;
			var foeLeft = far ? foeTotal - far.pokemon.filter(function (q) { return q.fainted; }).length : foeTotal;
			rows.push('<div class="pb-game" data-act="room" data-room="' + escHtml(id) + '" title="Open this battle">' +
				'<span class="pb-foe">vs ' + escHtml(far && far.name ? far.name : '?') + '</span>' +
				'<span class="pb-num">T' + Math.max(0, b.turn || 0) + '</span>' +
				'<span class="pb-num" title="Pokemon left: yours - theirs">' + (total ? mine : '?') + '-' + foeLeft + '</span>' +
				'<span class="pb-last">' + escHtml(st.lastWhy || '') + '</span></div>');
		});
		return rows;
	}

	function renderPanel() {
		if (!cfg.panel || typeof document === 'undefined' || !document.body) return;
		if (!ui.el) {
			try { buildPanel(); } catch (e) { return; }
		}
		var R = ui.refs;
		var put = function (key, html) {
			if (ui.cache[key] === html) return;
			ui.cache[key] = html;
			R[key].innerHTML = html;
		};
		var cls = function (node, name, on) { if (node.classList.contains(name) !== !!on) node.classList.toggle(name, !!on); };
		var sync = function (node, v) {
			if (node.type === 'checkbox') { if (node.checked !== !!v) node.checked = !!v; return; }
			if (document.activeElement !== node && node.value !== '' + v) node.value = '' + v;
		};

		cls(ui.el, 'pb-min', ui.collapsed);
		cls(R.dot, 'on', cfg.enabled);
		var games = record.wins + record.losses;
		var sr = null;
		try { sr = eloSeries(); } catch (e) {}
		put('rec', (sr ? 'Elo ' + sr.games[sr.games.length - 1].after + ' &middot; ' : '') + record.wins + 'W ' + record.losses + 'L' + (record.ties ? ' ' + record.ties + 'T' : '') +
			(games ? ' &middot; ' + Math.round(100 * record.wins / games) + '%' : ''));
		put('collapse', ui.collapsed ? '&#9650;' : '&#9660;');
		if (ui.collapsed) return;

		put('toggle', cfg.enabled ? 'Stop bot' : 'Start bot');
		cls(R.toggle, 'stop', cfg.enabled);
		cls(R.toggle, 'go', !cfg.enabled);
		put('queue', 'Auto-queue: ' + (cfg.autoSearch ? 'ON' : 'OFF'));
		cls(R.queue, 'on', cfg.autoSearch);
		put('status', statusText());

		var formats = randomFormats();
		var fkey = formats.map(function (f) { return f.id; }).join(',');
		if (fkey !== ui.formatsKey && document.activeElement !== R.format) {
			ui.formatsKey = fkey;
			R.format.innerHTML = formats.map(function (f) { return '<option value="' + escHtml(f.id) + '">' + escHtml(f.name) + '</option>'; }).join('');
		}
		sync(R.format, toID(cfg.format));
		sync(R.parallel, cfg.parallel);
		sync(R.maxGames, cfg.maxGames);
		sync(R.autoTimer, cfg.autoTimer);
		sync(R.closeFinished, cfg.closeFinished);
		sync(R.fast, cfg.fast);
		sync(R.dryRun, cfg.dryRun);

		var eloHtml = '';
		try { eloHtml = eloPanelHtml(sr); } catch (e2) {}
		put('elo', eloHtml || '<span class="pb-dim">No rated games yet. The rating and its graph appear after the first ladder game.</span>');

		var rows = liveRows();
		put('liveH', 'Live battles' + (rows.length ? ' (' + rows.length + ')' : ''));
		var onDisk = typeof W.psbotSave === 'function';
		put('recInfo', !cfg.record ? 'Recording is off.' :
			(onDisk ? recorder.toDisk + ' saved this session to the battles folder next to PSBot.exe (' + recorder.saved + ' older ones in this browser)' : recorder.saved + ' in this browser. Use export to get them as a file') +
			(recorder.error ? '<br><span class="pb-down">' + escHtml(recorder.error) + '</span>' : ''));
		put('live', rows.join('') || '<span class="pb-dim">None right now.</span>');

		if (ui.shownLogVer !== ui.logVer) {
			ui.shownLogVer = ui.logVer;
			var box = R.log;
			var stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 8;
			box.textContent = logBuf.slice(-80).join('\n');
			if (stick) box.scrollTop = box.scrollHeight;
		}
	}

	// ------------------------------------------------------------------ server popups

	// The server answers a refused search with a popup. Read the ones about its limits, adjust,
	// and keep them off the screen. Everything else is shown as usual.
	function onServerPopup(msg) {
		var m = /limited to (\d+) games at the same time/i.exec(msg);
		if (m) {
			if (Date.now() - lastSearchAt < 8000) record.searches = Math.max(queueBase, record.searches - 1);
			ui.serverCap = Math.max(1, lastSearchLive || +m[1]);
			ui.blockedUntil = Date.now() + 20000;
			log('This server allows ' + ui.serverCap + ' games at once. Holding at ' + ui.serverCap + '.');
			return true;
		}
		if (/limited to \d+ battles/i.test(msg)) {
			if (Date.now() - lastSearchAt < 8000) record.searches = Math.max(queueBase, record.searches - 1);
			ui.blockedUntil = Date.now() + 45000;
			log('Server search limit reached. Trying again in 45 seconds.');
			return true;
		}
		if (/already searching/i.test(msg)) return true;
		// The server is about to restart or has paused new battles: wait it out.
		if (/restart|lockdown|disabled|maintenance|try again later|too many/i.test(msg)) {
			ui.blockedUntil = Date.now() + 3 * 60000;
			event('server-notice', msg.slice(0, 200));
			return !!cfg.autoSearch;
		}
		if (cfg.autoSearch) event('popup', msg.slice(0, 200));
		return false;
	}
	function hookPopups() {
		var app = W.app;
		if (!app || app.__psbotAddPopup || typeof app.addPopup !== 'function') return;
		var orig = app.addPopup;
		app.__psbotAddPopup = orig;
		app.addPopup = function (type, data) {
			try {
				if (cfg.enabled && data && typeof data.message === 'string' && onServerPopup(data.message)) return null;
			} catch (e) {}
			return orig.apply(this, arguments);
		};
	}
	function unhookPopups() {
		var app = W.app;
		if (app && app.__psbotAddPopup) {
			app.addPopup = app.__psbotAddPopup;
			delete app.__psbotAddPopup;
		}
	}

	// Browsers throttle setInterval in background tabs to once a second, then once a minute.
	// A worker's timer is not throttled, so the bot keeps its pace when the tab is not in front.
	function startTicker(fn, ms) {
		try {
			var src = 'setInterval(function () { postMessage(0); }, ' + ms + ');';
			var url = URL.createObjectURL(new Blob([src], {type: 'application/javascript'}));
			var worker = new Worker(url);
			worker.onmessage = fn;
			return {stop: function () { worker.terminate(); URL.revokeObjectURL(url); }};
		} catch (e) {
			var id = setInterval(fn, ms);
			return {stop: function () { clearInterval(id); }};
		}
	}

	// ------------------------------------------------------------------ public API

	var api = {
		version: VERSION,
		config: cfg,
		record: record,
		elo: elo,
		predictions: predictions,
		records: {status: recorder, all: recAll, export: exportRecords},
		start: function (opts) {
			Object.assign(cfg, opts || {});
			cfg.parallel = clampInt(cfg.parallel, 1, MAX_PARALLEL);
			cfg.enabled = true;
			if (opts && opts.autoSearch) { queueBase = record.searches; lastSearchAt = 0; }
			hookPopups();
			var pf = toID(cfg.format), pg = /^gen(\d+)/.exec(pf);
			loadSets(pf, pg ? +pg[1] : 9);
			if (!timer) timer = startTicker(function () {
				try { tick(); } catch (e) { run.errors++; console.error('[PSBot] tick failed', e); }
			}, 300);
			log('Started' + (cfg.dryRun ? ' in dry-run mode' : '') + (cfg.autoSearch ? ', auto-queue ' + toID(cfg.format) : ''));
			saveRun();
			return 'PSBot running';
		},
		stop: function (silent) {
			// A hot update stops the old copy silently; that must not look like the user
			// switching the run off.
			if (!silent) { cfg.autoSearch = false; cfg.enabled = false; saveRun(); }
			cfg.enabled = false;
			cfg.autoSearch = false;
			if (timer) { timer.stop(); timer = null; }
			if (W.app && W.app.rooms) Object.keys(W.app.rooms).forEach(function (id) { unhookRoom(W.app.rooms[id]); });
			unhookPopups();
			if (silent) {
				if (ui.el && ui.el.parentNode) ui.el.parentNode.removeChild(ui.el);
				ui.el = null;
			} else {
				log('Stopped');
			}
			return 'PSBot stopped';
		},
		// Score table for the battle currently on screen, without sending anything.
		explain: function (roomid) {
			var room = roomid ? W.app.rooms[roomid] : W.app.curRoom;
			if (!room || !room.battle || !room.request) return lastExplain;
			var st = Object.assign({}, rooms[room.id] || {});
			var d = decide(room, room.request, st);
			return {choice: d.choice, why: d.why, options: (d.table || []).map(function (o) {
				return {option: o.label, score: +o.V.toFixed(3), ko: o.pKO === undefined ? '' : pct(o.pKO), risk: o.pDead === undefined ? '' : pct(o.pDead)};
			})};
		},
		log: function () { return logBuf.slice(); },
		_internals: {decide: decide, calc: calc, race: race, threat: threat, mkOpp: mkOpp, mkMine: mkMine, makeCtx: makeCtx, getMove: getMove, rooms: rooms,
			onRoomData: onRoomData, noteChoice: noteChoice, loadSets: loadSets, setsLoading: setsLoading,
			loadMoves: loadMoves, movesBusy: function () { return movesLoading; }, entryOf: entryOf, healOf: healOf, setupOf: setupOf}
	};
	api.build = BUILD;
	api.status = statusNow;
	// Everything a newer copy of the bot needs to carry on where this one stops.
	api._state = function () {
		return {rooms: rooms, record: record, predictions: predictions, queueBase: queueBase, eloStart: elo.sessionStart, lastSearchAt: lastSearchAt,
			run: run, cfg: clone(cfg), ui: {serverCap: ui.serverCap, blockedUntil: ui.blockedUntil}};
	};
	// Taking over from an older copy in the same page (a hot update): keep its games, its
	// counters and its switches.
	var prior = null;
	try { prior = W.__psbotCarry || null; W.__psbotCarry = null; } catch (e) {}
	if (prior && prior.rooms) {
		Object.keys(prior.rooms).forEach(function (rid) { rooms[rid] = prior.rooms[rid]; rooms[rid].synced = false; });
		['wins', 'losses', 'ties', 'games', 'searches'].forEach(function (k) { if (prior.record && typeof prior.record[k] === 'number') record[k] = prior.record[k]; });
		if (prior.predictions) Object.assign(predictions, prior.predictions);
		queueBase = prior.queueBase || 0;
		lastSearchAt = prior.lastSearchAt || 0;
		if (prior.eloStart !== undefined) elo.sessionStart = prior.eloStart;
		if (prior.run) { run.loadedAt = prior.run.loadedAt || run.loadedAt; run.reloads = prior.run.reloads || 0; run.reloadTimes = prior.run.reloadTimes || []; }
		if (prior.ui) { ui.serverCap = prior.ui.serverCap || 0; ui.blockedUntil = prior.ui.blockedUntil || 0; }
	}
	W.PSBot = api;
	loadMoves();
	sweepRecs();
	loadSettings();
	recCount();
	renderPanel();
	console.log('[PSBot] ' + VERSION + ' (' + BUILD + ') loaded. PSBot.start() to play, PSBot.start({dryRun: true}) to watch it think.');
	if (prior && prior.cfg) {
		// Same switches as the copy this one replaces.
		var was = prior.cfg;
		['format', 'parallel', 'maxGames', 'fast', 'dryRun', 'autoTimer', 'closeFinished', 'record'].forEach(function (k) { if (was[k] !== undefined) cfg[k] = was[k]; });
		if (was.enabled) { api.start(); cfg.autoSearch = !!was.autoSearch; saveRun(); }
		event('hot-update', {from: prior.version || '?', to: VERSION + ' ' + BUILD, games: Object.keys(rooms).length, auto: cfg.autoSearch});
	} else {
		// A page that reloaded in the middle of an unattended run picks the run back up.
		var last = loadRun();
		if (last && last.auto && Date.now() - last.ts < cfg.resumeMinutes * 60000) {
			['wins', 'losses', 'ties', 'games', 'searches'].forEach(function (k) { if (last.record && typeof last.record[k] === 'number') record[k] = last.record[k]; });
			queueBase = last.queueBase || 0;
			if (last.eloStart !== undefined && last.eloStart !== null) elo.sessionStart = last.eloStart;
			run.reloads = last.reloads || 0;
			run.reloadTimes = last.reloadTimes || [];
			if (last.predictions) Object.assign(predictions, last.predictions);
			run.resumed = true;
			setTimeout(function () {
				if (W.PSBot !== api) return;
				api.start({format: last.format, parallel: last.parallel, maxGames: last.maxGames});
				cfg.autoSearch = true;
				lastSearchAt = Date.now() - searchGap() + 8000; // give the login and the rejoined games a moment
				saveRun();
				event('resumed', {after: Math.round((Date.now() - last.ts) / 1000) + 's', record: record.wins + 'W ' + record.losses + 'L', reloads: run.reloads});
			}, 2500);
		}
	}
	// If the ticker ever stops (a worker killed by the browser), start it again.
	var guard = setInterval(function () {
		if (W.PSBot !== api || !cfg.enabled) return;
		if (Date.now() - run.lastTickAt > 15000) {
			run.errors++;
			if (timer) { try { timer.stop(); } catch (e) {} }
			timer = startTicker(function () { try { tick(); } catch (e2) { run.errors++; console.error('[PSBot] tick failed', e2); } }, 300);
			event('ticker-restarted', 'the tick loop had stopped');
		}
	}, 5000);
	if (guard && typeof guard.unref === 'function') guard.unref(); // under Node, do not hold the process open
})();
