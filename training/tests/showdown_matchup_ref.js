#!/usr/bin/env node
// What Showdown itself decides about a move against a Pokemon, for tests/test_matchup.py.
//
//   node tests/showdown_matchup_ref.js cases.jsonl > results.jsonl      (PS_DIR defaults to ~/pokemon-showdown)
//
// For every case a real one-against-one battle is built in the engine (dist/sim), the field is set, the
// Pokemon Terastallize, and the move goes through the same steps as sim/battle-actions.ts:
//   useMoveInner:      singleEvent ModifyType / ModifyMove of the move, then runEvent ModifyType / ModifyMove
//   trySpreadMoveHit:  runEvent TryHit (abilities such as Volt Absorb, Wonder Guard), then
//                      hitStepTypeImmunity (pokemon.runImmunity, with its default ignoreImmunity)
//   getDamage:         pokemon.runEffectiveness (the typeMod behind "-supereffective" / "-resisted")
// The move's own TryImmunity / Try handlers (Endeavor, Sucker Punch...) and accuracy are not run: they
// are not type matters (see matchup.py).
//
// Each output line: the engine's verdict plus the state the Pokemon ended up in after the battle started
// (species after Tera Shift / Terastallization, ability after Trace, item, volatiles, weather), which is
// what the Python side is given.
'use strict';

const fs = require('fs');
const path = require('path');
const PS = process.env.PS_DIR || path.join(process.env.HOME || '', 'pokemon-showdown');
const {Battle, Dex} = require(path.join(PS, 'dist', 'sim'));

function mkset(p, move) {
	const species = Dex.species.get(p.species);
	return {
		name: species.baseSpecies, species: p.species, item: p.item || '', ability: p.ability || '',
		moves: [move], nature: 'Serious', gender: '', level: 80,
		evs: {hp: 84, atk: 84, def: 84, spa: 84, spd: 84, spe: 84},
		ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
		teraType: p.tera || species.types[0],
	};
}

function state(mon) {
	return {
		species: mon.species.name,
		ability: mon.getAbility().name || '',
		item: mon.getItem().exists ? mon.getItem().name : '',
		tera: mon.terastallized || null,
		types: mon.getTypes(),
		volatiles: Object.keys(mon.volatiles),
		full_hp: mon.hp === mon.maxhp,
	};
}

function run(c) {
	const battle = new Battle({formatid: 'gen9randombattle'});
	battle.setPlayer('p1', {name: 'A', team: [mkset(c.atk, c.move)]});
	battle.setPlayer('p2', {name: 'B', team: [mkset(c.def, 'splash')]});
	const atk = battle.p1.active[0];
	const def = battle.p2.active[0];
	// Switch-in abilities may have set weather or terrain: the case decides the field.
	battle.field.clearWeather();
	battle.field.clearTerrain();
	if (c.weather) battle.field.setWeather(c.weather, 'debug');
	if (c.terrain) battle.field.setTerrain(c.terrain, 'debug');
	if (c.gravity) battle.field.addPseudoWeather('gravity', 'debug');
	if (c.atk.tera) battle.actions.terastallize(atk);
	if (c.def.tera) battle.actions.terastallize(def);
	for (const v of c.def_volatiles || []) def.addVolatile(v);

	const move = battle.dex.getActiveMove(c.move);
	battle.setActiveMove(move, atk, def);
	battle.singleEvent('ModifyType', move, null, atk, def, move, move);
	battle.singleEvent('ModifyMove', move, null, atk, def, move, move);
	battle.runEvent('ModifyType', atk, def, move, move);
	battle.runEvent('ModifyMove', atk, def, move, move);
	const moveType = move.type;

	let immune = false;
	let by = '';
	const tryHit = battle.runEvent('TryHit', def, atk, move);
	if (!tryHit && tryHit !== 0) {
		immune = true;
		by = 'tryhit';
	}
	if (!immune) {
		if (move.ignoreImmunity === undefined) move.ignoreImmunity = (move.category === 'Status');
		if (!def.runImmunity(move)) {
			immune = true;
			by = 'type';
		}
	}
	let typeMod = 0;
	if (!immune) typeMod = battle.clampIntRange(def.runEffectiveness(move), -6, 6);
	const fixed = !!(move.ohko || move.damageCallback || move.damage);
	return {
		id: c.id, move_type: moveType, category: move.category, immune, by, type_mod: typeMod, fixed,
		atk: state(atk), def: state(def),
		weather: battle.field.weather || null, terrain: battle.field.terrain || null,
		gravity: !!battle.field.pseudoWeather['gravity'],
	};
}

const input = process.argv[2] ? fs.readFileSync(process.argv[2], 'utf8') : fs.readFileSync(0, 'utf8');
const out = [];
for (const line of input.split('\n')) {
	if (!line.trim()) continue;
	const c = JSON.parse(line);
	try {
		out.push(JSON.stringify(run(c)));
	} catch (e) {
		out.push(JSON.stringify({id: c.id, error: String(e && e.stack || e).slice(0, 400)}));
	}
}
process.stdout.write(out.join('\n') + '\n');
