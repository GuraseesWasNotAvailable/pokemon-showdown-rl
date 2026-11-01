#!/usr/bin/env node
// Downloads what the headless engine needs into heuristic/cache: the Showdown client's battle
// code and data, and the random-battle set data. The client code is AGPL and belongs to the
// Showdown project, so it is fetched here instead of being committed to this repo.
//
//   node heuristic/setup.js              gen 9 only
//   node heuristic/setup.js --all-gens   also the 15 MB table older generations need
//   node heuristic/setup.js --force      download again even if a file is already there
'use strict';

const fs = require('fs');
const path = require('path');
const {CLIENT_FILES, OLD_GEN_TABLES, DEFAULT_CACHE} = require('./engine');

const CLIENT = 'https://play.pokemonshowdown.com/';
const SETS = 'https://pkmn.github.io/randbats/data/';
const JS_FILES = new Set(['battle-dex-data.js', 'battle-dex.js', 'battle-text-parser.js', 'battle-scene-stub.js', 'battle.js']);
const SET_FILES = ['gen9randombattle.json'];

(async () => {
	const args = new Set(process.argv.slice(2));
	const cache = process.env.PSBOT_CACHE || DEFAULT_CACHE;
	fs.mkdirSync(cache, {recursive: true});
	const jobs = CLIENT_FILES.map(f => [f, CLIENT + (JS_FILES.has(f) ? 'js/' : 'data/') + f]);
	if (args.has('--all-gens')) jobs.push([OLD_GEN_TABLES, CLIENT + 'data/' + OLD_GEN_TABLES]);
	for (const f of SET_FILES) jobs.push([f, SETS + f]);
	// The full move data. The bot keeps its own copy because the client strips moves as it uses them.
	jobs.push(['moves.json', CLIENT + 'data/moves.json']);

	let failed = 0;
	for (const [name, url] of jobs) {
		const file = path.join(cache, name);
		if (fs.existsSync(file) && !args.has('--force')) {
			console.log(`have   ${name}`);
			continue;
		}
		try {
			const res = await fetch(url);
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			const buf = Buffer.from(await res.arrayBuffer());
			fs.writeFileSync(file, buf);
			console.log(`saved  ${name} (${Math.round(buf.length / 1024)} KB)`);
		} catch (e) {
			failed++;
			console.error(`FAILED ${name}: ${e.message}`);
		}
	}
	if (failed) process.exit(1);
	console.log(`\nCache ready in ${cache}`);
})();
