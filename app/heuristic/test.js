#!/usr/bin/env node
// Runs the PSBot position tests (psbot/pstest.js) headless, through the same engine the Python
// bridge uses. Scenarios from generations before 9 need teambuilder-tables.js in the cache
// (node heuristic/setup.js --all-gens) and are skipped without it.
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {createEngine} = require('./engine');

(async () => {
	const engine = createEngine({log: () => {}});
	const file = path.join(__dirname, '..', 'psbot', 'pstest.js');
	vm.runInContext(fs.readFileSync(file, 'utf8'), engine.sandbox, {filename: file});
	const T = engine.sandbox.PSTest;
	if (!engine.oldGens) {
		const before = T.scenarios.length;
		for (let i = T.scenarios.length - 1; i >= 0; i--) {
			if ((T.scenarios[i].gen || 9) < 9) T.scenarios.splice(i, 1);
		}
		console.log(`(skipping ${before - T.scenarios.length} old-generation scenarios: no ${'teambuilder-tables.js'} in the cache)`);
	}
	const res = await T.run();
	let pass = 0;
	for (const r of res) {
		if (r.pass) pass++;
		console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}${r.pass ? '' : `\n      got ${r.got}\n      want ${r.expect}\n      top ${r.top}`}`);
	}
	console.log(`\n${pass}/${res.length} position tests pass`);
	process.exit(pass === res.length ? 0 : 1);
})().catch(e => { console.error(e); process.exit(2); });
