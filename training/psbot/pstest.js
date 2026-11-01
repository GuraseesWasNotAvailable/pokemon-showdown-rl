// Position tests for PSBot. Each scenario rebuilds a real position in a headless Battle,
// hands the bot a matching request, and checks its choice against the known-good move.
// Load in a Showdown page after psbot.js:  await PSTest.run()
(function () {
	'use strict';
	var toID = function (s) { return ('' + s).toLowerCase().replace(/[^a-z0-9]+/g, ''); };

	function statsFor(species, level, gen) {
		var sp = Dex.forGen(gen).species.get(species);
		var b = sp.baseStats;
		var term = gen >= 3 ? 52 : 93;
		var f = function (x) { return Math.floor((2 * x + term) * level / 100) + 5; };
		return {
			hp: b.hp === 1 ? 1 : Math.floor((2 * b.hp + term) * level / 100) + level + 10,
			stats: {atk: f(b.atk), def: f(b.def), spa: f(b.spa), spd: f(b.spd), spe: f(b.spe)}
		};
	}

	function build(sc) {
		var gen = sc.gen || 9;
		var fmt = sc.fmt || ('gen' + gen + 'randombattle');
		var lines = ['|player|p1|Me|1|', '|player|p2|Foe|2|', '|teamsize|p1|' + sc.mine.length, '|teamsize|p2|' + (sc.foeTotal || 6),
			'|gametype|singles', '|gen|' + gen, '|tier|[Gen ' + gen + '] Random Battle', '|start'];
		var mons = sc.mine.map(function (m, i) {
			var st = statsFor(m.species, m.level, gen);
			var hp = Math.max(1, Math.round(st.hp * (m.hp === undefined ? 1 : m.hp)));
			return {
				ident: 'p1: ' + m.species, details: m.species + ', L' + m.level,
				condition: hp + '/' + st.hp + (m.status ? ' ' + m.status : ''), active: i === 0,
				stats: st.stats, moves: m.moves.map(toID), baseAbility: toID(m.ability || ''), ability: toID(m.ability || ''),
				item: toID(m.item || ''), pokeball: 'pokeball', teraType: m.tera || '', terastallized: m.terastallized || '', _hp: hp, _max: st.hp
			};
		});
		// Reveal the foe's bench first, then its active.
		(sc.foeBench || []).forEach(function (f) {
			lines.push('|switch|p2a: ' + f.species + '|' + f.species + ', L' + f.level + '|' + Math.round((f.hp === undefined ? 1 : f.hp) * 100) + '/100');
		});
		lines.push('|switch|p1a: ' + sc.mine[0].species + '|' + mons[0].details + '|' + mons[0].condition);
		lines.push('|switch|p2a: ' + sc.foe.species + '|' + sc.foe.species + ', L' + sc.foe.level + '|' + Math.round((sc.foe.hp === undefined ? 1 : sc.foe.hp) * 100) + '/100' + (sc.foe.status ? ' ' + sc.foe.status : ''));
		(sc.lines || []).forEach(function (l) { lines.push(l); });
		lines.push('|turn|' + (sc.turn || 5));

		// The live client only ships the animated scene, so give it detached elements to draw
		// into. Under Node (heuristic/engine.js) the stub scene is loaded and no DOM is needed.
		var bopts = {id: 'battle-' + fmt + '-9999', paused: true};
		if (typeof BattleSceneStub === 'undefined') {
			bopts.$frame = $('<div class="battle"></div>');
			bopts.$logFrame = $('<div class="battle-log"></div>');
		}
		var battle = new Battle(bopts);
		lines.forEach(function (l) { battle.stepQueue.push(l); });
		battle.seekTurn(Infinity, true);

		var act = {moves: sc.mine[0].moves.map(function (name) {
			return {move: name, id: toID(name), pp: 16, maxpp: 16, target: 'normal', disabled: (sc.disabled || []).indexOf(toID(name)) >= 0};
		})};
		if (sc.mine[0].tera && !sc.mine[0].terastallized && gen === 9 && !sc.noTera) act.canTerastallize = sc.mine[0].tera;
		if (sc.canDynamax) act.canDynamax = true;
		if (sc.canMegaEvo) act.canMegaEvo = true;
		var req = {active: [act], side: {name: 'Me', id: 'p1', pokemon: mons}, rqid: 7};
		if (sc.forceSwitch) req = {forceSwitch: [true], side: req.side, rqid: 7};
		var room = {id: 'battle-' + fmt + '-9999', battle: battle, request: req, side: 'p1'};
		return {room: room, req: req, battle: battle, fmt: fmt, gen: gen};
	}

	var SCENARIOS = [
		{name: 'Kingambit +2 vs Gallade: priority before the 4x Close Combat', expect: ['Sucker Punch'],
			mine: [{species: 'Kingambit', level: 74, moves: ['Iron Head', 'Swords Dance', 'Sucker Punch', 'Kowtow Cleave'], ability: 'Supreme Overlord', item: 'Leftovers', tera: 'Flying'},
				{species: 'Qwilfish', level: 86, moves: ['Waterfall', 'Toxic Spikes', 'Spikes', 'Thunder Wave'], ability: 'Intimidate', item: 'Rocky Helmet'},
				{species: 'Chimecho', level: 93, moves: ['Dazzling Gleam', 'Psychic Noise', 'Calm Mind', 'Recover'], ability: 'Levitate', item: 'Leftovers'}],
			foe: {species: 'Gallade', level: 80}, lines: ['|-boost|p1a: Kingambit|atk|2', '|-start|p1a: Kingambit|fallen2|[silent]'], noTera: true},
		{name: 'Politoed vs Solgaleo in rain: Toxic is immune, Scald is the hit', expect: ['Scald'],
			mine: [{species: 'Politoed', level: 86, moves: ['Scald', 'Ice Beam', 'Toxic', 'Protect'], ability: 'Drizzle', item: 'Leftovers'},
				{species: 'Togekiss', level: 82, moves: ['Air Slash', 'Nasty Plot', 'Roost', 'Aura Sphere'], ability: 'Serene Grace', item: 'Leftovers'}],
			gen: 8, foe: {species: 'Solgaleo', level: 71}, lines: ['|-weather|RainDance|[from] ability: Drizzle|[of] p1a: Politoed']},
		{name: 'Koraidon vs +2 Rillaboom in sun: Flare Blitz kills', expect: ['Flare Blitz'],
			mine: [{species: 'Koraidon', level: 64, moves: ['Close Combat', 'Flare Blitz', 'U-turn', 'Outrage'], ability: 'Orichalcum Pulse', item: 'Choice Band', tera: 'Fire'},
				{species: 'Clefable', level: 83, moves: ['Calm Mind', 'Fire Blast', 'Moonlight', 'Moonblast'], ability: 'Magic Guard', item: 'Leftovers'}],
			foe: {species: 'Rillaboom', level: 79}, lines: ['|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge|[of] p2a: Rillaboom', '|-weather|SunnyDay|[from] ability: Orichalcum Pulse|[of] p1a: Koraidon', '|move|p2a: Rillaboom|Swords Dance|p2a: Rillaboom', '|-boost|p2a: Rillaboom|atk|2']},
		{name: 'Plusle vs Rillaboom turn 1: get out', expect: ['switch Koraidon', 'switch Entei'],
			mine: [{species: 'Plusle', level: 95, moves: ['Encore', 'Alluring Voice', 'Nasty Plot', 'Thunderbolt'], ability: 'Lightning Rod', item: 'Life Orb', tera: 'Electric'},
				{species: 'Koraidon', level: 64, moves: ['Close Combat', 'Flare Blitz', 'U-turn', 'Outrage'], ability: 'Orichalcum Pulse', item: 'Choice Band'},
				{species: 'Entei', level: 78, moves: ['Sacred Fire', 'Extreme Speed', 'Stomping Tantrum', 'Flare Blitz'], ability: 'Inner Focus', item: 'Choice Band'},
				{species: 'Cacturne', level: 92, moves: ['Knock Off', 'Sucker Punch', 'Spikes', 'Leaf Storm'], ability: 'Water Absorb', item: 'Life Orb'}],
			foe: {species: 'Rillaboom', level: 79}, turn: 1, lines: ['|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge|[of] p2a: Rillaboom']},
		{name: 'Mismagius vs Tornadus: Thunderbolt, or Nasty Plot first', expect: ['Thunderbolt', 'Nasty Plot'],
			mine: [{species: 'Mismagius', level: 86, moves: ['Shadow Ball', 'Thunderbolt', 'Nasty Plot', 'Substitute'], ability: 'Levitate', item: 'Leftovers', tera: 'Electric'},
				{species: 'Magcargo', level: 95, moves: ['Power Gem', 'Stealth Rock', 'Lava Plume', 'Recover'], ability: 'Flame Body', item: 'Heavy-Duty Boots'}],
			foe: {species: 'Tornadus', level: 79}, turn: 1},
		{name: 'Clefable vs Tera Steel Haxorus locked into Outrage: Fire Blast', expect: ['Fire Blast'],
			mine: [{species: 'Clefable', level: 83, moves: ['Calm Mind', 'Fire Blast', 'Moonlight', 'Moonblast'], ability: 'Magic Guard', item: 'Leftovers', tera: 'Steel'},
				{species: 'Entei', level: 78, moves: ['Sacred Fire', 'Extreme Speed', 'Stomping Tantrum', 'Flare Blitz'], ability: 'Inner Focus', item: 'Choice Band'}],
			foe: {species: 'Haxorus', level: 78, hp: 0.71}, noTera: true,
			lines: ['|-weather|SunnyDay', '|-terastallize|p2a: Haxorus|Steel', '|move|p2a: Haxorus|Outrage|p1a: Clefable', '|-immune|p1a: Clefable']},
		{name: 'Chimecho +2 vs Bisharp at 40%: Psychic Noise is immune, Dazzling Gleam kills', expect: ['Dazzling Gleam'],
			mine: [{species: 'Chimecho', level: 93, moves: ['Dazzling Gleam', 'Psychic Noise', 'Calm Mind', 'Recover'], ability: 'Levitate', item: 'Leftovers'},
				{species: 'Qwilfish', level: 86, moves: ['Waterfall', 'Toxic Spikes', 'Spikes', 'Thunder Wave'], ability: 'Intimidate', item: 'Rocky Helmet'}],
			foe: {species: 'Bisharp', level: 79, hp: 0.4}, lines: ['|-boost|p1a: Chimecho|spa|2', '|-boost|p1a: Chimecho|spd|2', '|move|p2a: Bisharp|Sucker Punch|p1a: Chimecho', '|move|p2a: Bisharp|Throat Chop|p1a: Chimecho'], noTera: true},
		{name: 'Milotic vs burned Venusaur: Ice Beam, not Scald (or hand over to Charizard)', expect: ['Ice Beam', 'switch Charizard'],
			mine: [{species: 'Milotic', level: 82, moves: ['Scald', 'Haze', 'Recover', 'Ice Beam'], ability: 'Marvel Scale', item: 'Leftovers', hp: 0.93},
				{species: 'Charizard', level: 85, moves: ['Air Slash', 'Fire Blast', 'Roost', 'Earthquake'], ability: 'Blaze', item: 'Heavy-Duty Boots'}],
			gen: 7, foe: {species: 'Venusaur', level: 84, hp: 0.76, status: 'brn'}},
		{name: 'Ho-Oh at 10% vs Darmanitan with Sudowoodo in the back', expect: ['Brave Bird', 'switch Sudowoodo'],
			mine: [{species: 'Ho-Oh', level: 69, moves: ['Sacred Fire', 'Toxic', 'Recover', 'Brave Bird'], ability: 'Regenerator', item: 'Heavy-Duty Boots', hp: 0.1},
				{species: 'Sudowoodo', level: 90, moves: ['Earthquake', 'Head Smash', 'Stealth Rock', 'Wood Hammer'], ability: 'Rock Head', item: 'Choice Band'}],
			gen: 8, foe: {species: 'Darmanitan', level: 78}, foeTotal: 6, lines: ['|-weather|RainDance']},
		{name: 'Stakataka vs +1 Haxorus: Gyro Ball', expect: ['Gyro Ball', 'Max Gyro Ball'],
			mine: [{species: 'Stakataka', level: 81, moves: ['Gyro Ball', 'Stone Edge', 'Body Press', 'Trick Room'], ability: 'Beast Boost', item: 'Life Orb'},
				{species: 'Kingdra', level: 83, moves: ['Rain Dance', 'Hydro Pump', 'Draco Meteor', 'Hurricane'], ability: 'Swift Swim', item: 'Life Orb'}],
			gen: 8, canDynamax: true, foe: {species: 'Haxorus', level: 77}, lines: ['|move|p2a: Haxorus|Dragon Dance|p2a: Haxorus', '|-boost|p2a: Haxorus|atk|1', '|-boost|p2a: Haxorus|spe|1']},
		{name: 'Gen 1 Tauros vs Goldeen: Body Slam over Hyper Beam', expect: ['Body Slam'],
			mine: [{species: 'Tauros', level: 68, moves: ['Earthquake', 'Blizzard', 'Body Slam', 'Hyper Beam']},
				{species: 'Jynx', level: 73, moves: ['Blizzard', 'Lovely Kiss', 'Psychic', 'Rest']}],
			gen: 1, foe: {species: 'Goldeen', level: 88}, turn: 1},
		{name: 'Gen 1 Seadra vs Kabuto at 52%: Surf', expect: ['Surf'],
			mine: [{species: 'Seadra', level: 77, moves: ['Agility', 'Surf', 'Hyper Beam', 'Blizzard'], hp: 0.8},
				{species: 'Jynx', level: 73, moves: ['Blizzard', 'Lovely Kiss', 'Psychic', 'Rest']}],
			gen: 1, foe: {species: 'Kabuto', level: 88, hp: 0.52}},
		{name: 'Metagross-Mega at -1 vs Florges at 68%: Meteor Mash', expect: ['Meteor Mash'],
			mine: [{species: 'Metagross-Mega', level: 45, moves: ['Psychic Fangs', 'Iron Defense', 'Meteor Mash', 'Body Press'], ability: 'Tough Claws', item: 'Metagrossite'},
				{species: 'Gogoat', level: 55, moves: ['Horn Leech', 'Milk Drink', 'Earthquake', 'Bulk Up'], ability: 'Sap Sipper', item: 'Leftovers'}],
			fmt: 'gen9championsrandombattle', foe: {species: 'Florges', level: 54, hp: 0.68}, lines: ['|-unboost|p1a: Metagross|atk|1'], noTera: true},
		{name: 'Forced switch into Medicham threat: Plusle outspeeds and kills', expect: ['switch Plusle', 'switch Clefable'], forceSwitch: true,
			mine: [{species: 'Cacturne', level: 92, moves: ['Knock Off', 'Sucker Punch', 'Spikes', 'Leaf Storm'], ability: 'Water Absorb', item: 'Life Orb', hp: 0},
				{species: 'Plusle', level: 95, moves: ['Encore', 'Alluring Voice', 'Nasty Plot', 'Thunderbolt'], ability: 'Lightning Rod', item: 'Life Orb'},
				{species: 'Clefable', level: 83, moves: ['Calm Mind', 'Fire Blast', 'Moonlight', 'Moonblast'], ability: 'Magic Guard', item: 'Leftovers', hp: 0.34},
				{species: 'Decidueye-Hisui', level: 88, moves: ['Leaf Blade', 'Triple Arrows', 'Sucker Punch', 'Swords Dance'], ability: 'Scrappy', item: 'Life Orb'}],
			foe: {species: 'Medicham', level: 86, hp: 0.57}, lines: ['|move|p2a: Medicham|Bullet Punch|p1a: Cacturne']},
		{name: 'Garchomp vs Kingambit that Terastallized into Flying: Earthquake no longer hits', expect: ['Swords Dance', 'Stone Edge', 'Scale Shot'], avoid: ['Earthquake'],
			mine: [{species: 'Garchomp', level: 74, moves: ['Earthquake', 'Stone Edge', 'Scale Shot', 'Swords Dance'], ability: 'Rough Skin', item: 'Loaded Dice', tera: 'Fire'},
				{species: 'Clefable', level: 83, moves: ['Calm Mind', 'Fire Blast', 'Moonlight', 'Moonblast'], ability: 'Magic Guard', item: 'Leftovers'}],
			foe: {species: 'Kingambit', level: 74}, noTera: true, lines: ['|-terastallize|p2a: Kingambit|Flying']},
		{name: 'Volcarona vs Terrakion with Stone Edge: Tera Ground turns the 4x hit into a resisted one (optional defensive Tera)', expectTera: true,
			cfg: {teraSurprise: true, teraStatus: true},
			mine: [{species: 'Volcarona', level: 77, moves: ['Fiery Dance', 'Bug Buzz', 'Giga Drain', 'Quiver Dance'], ability: 'Flame Body', item: 'Heavy-Duty Boots', tera: 'Ground'}],
			foe: {species: 'Terrakion', level: 79}, lines: ['|move|p2a: Terrakion|Stone Edge|p1a: Volcarona|[miss]', '|-miss|p2a: Terrakion|p1a: Volcarona']},
		{name: 'Dragonite at 60% vs Weavile at 40%: Tera Normal Extreme Speed takes the KO first', expect: ['Extreme Speed +Tera Normal'], expectTera: true,
			mine: [{species: 'Dragonite', level: 74, moves: ['Extreme Speed', 'Earthquake', 'Dragon Dance', 'Outrage'], ability: 'Multiscale', item: 'Heavy-Duty Boots', tera: 'Normal', hp: 0.6},
				{species: 'Clefable', level: 83, moves: ['Calm Mind', 'Fire Blast', 'Moonlight', 'Moonblast'], ability: 'Magic Guard', item: 'Leftovers'}],
			foe: {species: 'Weavile', level: 79, hp: 0.4}, lines: ['|move|p2a: Weavile|Triple Axel|p1a: Dragonite']},
		{name: 'Garchomp vs Toxtricity with Corviknight in the back, against an opponent who has been switching: expect the Ground immunity', predict: 'Corviknight',
			st: {oppSwitchTurns: [3, 4]},
			mine: [{species: 'Garchomp', level: 74, moves: ['Earthquake', 'Stone Edge', 'Scale Shot', 'Swords Dance'], ability: 'Rough Skin', item: 'Loaded Dice', tera: 'Fire'},
				{species: 'Clefable', level: 83, moves: ['Calm Mind', 'Fire Blast', 'Moonlight', 'Moonblast'], ability: 'Magic Guard', item: 'Leftovers'}],
			foe: {species: 'Toxtricity', level: 83}, foeBench: [{species: 'Corviknight', level: 80}, {species: 'Blissey', level: 85}], noTera: true},
		{name: 'Blastoise vs a frozen Ceruledge with Latias in the back: hit it or set up, do not Ice Beam the switch that is not coming', expect: ['Earthquake', 'Hydro Pump', 'Shell Smash'], avoid: ['Ice Beam'],
			mine: [{species: 'Blastoise', level: 80, moves: ['Ice Beam', 'Hydro Pump', 'Shell Smash', 'Earthquake'], ability: 'Torrent', item: 'White Herb', tera: 'Water', hp: 0.58},
				{species: 'Klawf', level: 90, moves: ['Knock Off', 'Stealth Rock', 'Crabhammer', 'Stone Edge'], ability: 'Regenerator', item: 'Life Orb'}],
			foe: {species: 'Ceruledge', level: 78, hp: 0.71, status: 'frz'}, foeBench: [{species: 'Latias', level: 79, hp: 0.55}, {species: 'Bisharp', level: 79}], noTera: true, turn: 14,
			st: {oppSwitchTurns: [9, 12]}, lines: ['|move|p2a: Ceruledge|Poltergeist|p1a: Blastoise']},
		{name: 'Chimecho at 19% vs Iron Hands at 9%: take the KO instead of healing again', expect: ['Psychic Noise'],
			mine: [{species: 'Chimecho', level: 93, moves: ['Encore', 'Recover', 'Psychic Noise', 'Heal Bell'], ability: 'Levitate', item: 'Leftovers', hp: 0.19},
				{species: 'Hawlucha', level: 80, moves: ['Swords Dance', 'Stone Edge', 'Acrobatics', 'Close Combat'], ability: 'Unburden', item: 'White Herb'}],
			foe: {species: 'Iron Hands', level: 79, hp: 0.09}, foeBench: [{species: 'Arceus-Ghost', level: 69, hp: 0.67}], noTera: true, turn: 27,
			lines: ['|-terastallize|p2a: Iron Hands|Electric', '|move|p2a: Iron Hands|Thunder Punch|p1a: Chimecho', '|move|p2a: Iron Hands|Heavy Slam|p1a: Chimecho']},
		{name: 'Mightyena locked into Throat Chop vs Golem-Alola: go down swinging so Stantler comes in free, or hand over now', expect: ['Throat Chop', 'switch Stantler'],
			mine: [{species: 'Mightyena', level: 96, moves: ['Poison Fang', 'Throat Chop', 'Sucker Punch', 'Play Rough'], ability: 'Intimidate', item: 'Choice Band', hp: 0.94},
				{species: 'Stantler', level: 90, moves: ['Thunder Wave', 'Earthquake', 'Throat Chop', 'Body Slam'], ability: 'Intimidate', item: 'Eviolite'},
				{species: 'Amoonguss', level: 82, moves: ['Stomping Tantrum', 'Spore', 'Sludge Bomb', 'Giga Drain'], ability: 'Regenerator', item: 'Leftovers'}],
			foe: {species: 'Golem-Alola', level: 93, hp: 0.69}, noTera: true, turn: 3, disabled: ['poisonfang', 'suckerpunch', 'playrough'],
			lines: ['|move|p1a: Mightyena|Throat Chop|p2a: Golem', '|move|p2a: Golem|Rock Polish|p2a: Golem', '|-boost|p2a: Golem|spe|2', '|move|p2a: Golem|Earthquake|p1a: Mightyena']},
		{name: 'Blastoise at full health vs Toxapex with five behind it: Shell Smash is the way through', expect: ['Shell Smash'],
			mine: [{species: 'Blastoise', level: 80, moves: ['Ice Beam', 'Hydro Pump', 'Shell Smash', 'Earthquake'], ability: 'Torrent', item: 'White Herb', tera: 'Water'},
				{species: 'Klawf', level: 90, moves: ['Knock Off', 'Stealth Rock', 'Crabhammer', 'Stone Edge'], ability: 'Regenerator', item: 'Life Orb'}],
			foe: {species: 'Toxapex', level: 82}, foeBench: [{species: 'Garchomp', level: 74}, {species: 'Volcarona', level: 77}], noTera: true, turn: 4,
			lines: ['|move|p2a: Toxapex|Toxic Spikes|p1a: Blastoise']},
		{name: 'Sableye vs Cetitan that just used Belly Drum: Encore it or hit it, do not sit there healing', avoid: ['Recover'],
			mine: [{species: 'Sableye', level: 91, moves: ['Encore', 'Recover', 'Knock Off', 'Will-O-Wisp'], ability: 'Prankster', item: 'Leftovers', hp: 0.6},
				{species: 'Klawf', level: 90, moves: ['Knock Off', 'Stealth Rock', 'Crabhammer', 'Stone Edge'], ability: 'Regenerator', item: 'Life Orb'}],
			foe: {species: 'Cetitan', level: 82, hp: 0.75}, noTera: true, turn: 11,
			lines: ['|move|p2a: Cetitan|Belly Drum|p2a: Cetitan', '|-setboost|p2a: Cetitan|atk|6|[from] move: Belly Drum']},
		{name: 'Forced switch with Stealth Rock up: a Lugia on 15% with no Boots dies on the way in, so not that one', avoid: ['switch Lugia'], forceSwitch: true,
			mine: [{species: 'Wyrdeer', level: 87, moves: ['Thunder Wave', 'Earthquake', 'Psychic Noise', 'Megahorn'], ability: 'Intimidate', item: 'Leftovers', hp: 0},
				{species: 'Lugia', level: 73, moves: ['Calm Mind', 'Recover', 'Aeroblast', 'Earth Power'], ability: 'Multiscale', item: '', hp: 0.15},
				{species: 'Revavroom', level: 79, moves: ['Shift Gear', 'High Horsepower', 'Iron Head', 'Gunk Shot'], ability: 'Filter', item: 'Life Orb'},
				{species: 'Minior', level: 79, moves: ['Power Gem', 'Acrobatics', 'Earthquake', 'Shell Smash'], ability: 'Shields Down', item: 'White Herb', hp: 0.5}],
			foe: {species: 'Spectrier', level: 75, hp: 0.91}, foeTotal: 6,
			lines: ['|-sidestart|p1: Me|move: Stealth Rock', '|move|p2a: Spectrier|Shadow Ball|p1a: Wyrdeer']},
		{name: 'Smeargle with its Sash vs Espathra at +2 and four Speed boosts: blow it out or put it to sleep, not Sticky Web', expect: ['Whirlwind', 'Spore'],
			mine: [{species: 'Smeargle', level: 95, moves: ['Spore', 'Ceaseless Edge', 'Sticky Web', 'Whirlwind'], ability: 'Own Tempo', item: 'Focus Sash'},
				{species: 'Oricorio', level: 84, moves: ['Roost', 'Hurricane', 'Quiver Dance', 'Revelation Dance'], ability: 'Dancer', item: 'Heavy-Duty Boots'}],
			foe: {species: 'Espathra', level: 80, hp: 0.64}, foeBench: [{species: 'Gengar', level: 81, hp: 0.9}], noTera: true, turn: 17,
			lines: ['|move|p2a: Espathra|Calm Mind|p2a: Espathra', '|-boost|p2a: Espathra|spa|2', '|-boost|p2a: Espathra|spd|2', '|-boost|p2a: Espathra|spe|4',
				'|move|p2a: Espathra|Stored Power|p1a: Smeargle', '|move|p2a: Espathra|Protect|p2a: Espathra', '|move|p2a: Espathra|Dazzling Gleam|p1a: Smeargle']},
		{name: 'Magcargo vs Scrafty at 47% that has shown Bulk Up: hit it now, rocks and Recover only give it boosts', expect: ['Lava Plume'],
			mine: [{species: 'Magcargo', level: 95, moves: ['Recover', 'Stealth Rock', 'Lava Plume', 'Yawn'], ability: 'Flame Body', item: 'Heavy-Duty Boots', hp: 0.8},
				{species: 'Dunsparce', level: 86, moves: ['Body Slam', 'Coil', 'Earthquake', 'Roost'], ability: 'Serene Grace', item: 'Eviolite'}],
			foe: {species: 'Scrafty', level: 83, hp: 0.47}, foeBench: [{species: 'Reuniclus', level: 88}], foeTotal: 6, noTera: true, turn: 22,
			lines: ['|move|p2a: Scrafty|Bulk Up|p2a: Scrafty', '|move|p2a: Scrafty|Knock Off|p1a: Magcargo', '|move|p2a: Scrafty|Drain Punch|p1a: Magcargo']}
	];

	function runOne(sc) {
		var built = build(sc);
		if (sc.forceSwitch) built.req.side.pokemon[0].condition = '0 fnt';
		// A scenario can switch on an optional behaviour (cfg) or give the bot a history (st).
		var saved = {};
		for (var ck in sc.cfg || {}) { saved[ck] = PSBot.config[ck]; PSBot.config[ck] = sc.cfg[ck]; }
		var d;
		try {
			d = PSBot._internals.decide(built.room, built.req, JSON.parse(JSON.stringify(sc.st || {})));
		} finally {
			for (var rk in saved) PSBot.config[rk] = saved[rk];
		}
		var label = d.why.replace(/ vs .*$/, '');
		var pass = !sc.expect || sc.expect.some(function (e) { return label.indexOf(e) === 0; });
		if (sc.expect && sc.expect.indexOf('switch') >= 0) pass = /^switch /.test(d.choice);
		var want = sc.expect ? [sc.expect.join(' / ')] : [];
		if (sc.avoid) {
			pass = pass && !sc.avoid.some(function (e) { return label.indexOf(e) === 0; });
			want.push('not ' + sc.avoid.join(' / '));
		}
		if (sc.expectTera !== undefined) {
			pass = pass && / terastallize/.test(d.choice) === sc.expectTera;
			want.push(sc.expectTera ? 'with Tera' : 'without Tera');
		}
		if (sc.predict) {
			pass = pass && !!d.predict && d.predict.name === sc.predict;
			want.push('predicting a switch to ' + sc.predict);
		}
		return {
			name: sc.name, pass: pass, got: label + ' [' + d.choice + ']' + (d.predict ? ', predicts ' + d.predict.name : ''), expect: want.join(', '),
			top: (d.table || []).slice(0, 5).map(function (o) { return o.label + ' ' + o.V.toFixed(2); }).join(' | ')
		};
	}

	window.PSTest = {
		scenarios: SCENARIOS,
		build: build,
		runOne: runOne,
		run: async function () {
			// Set data loads asynchronously the first time a format is seen.
			SCENARIOS.forEach(function (sc) { try { var b = build(sc); PSBot._internals.makeCtx(b.room, b.req); } catch (e) {} });
			await new Promise(function (r) { setTimeout(r, 3500); });
			var out = SCENARIOS.map(function (sc) {
				try { return runOne(sc); } catch (e) { return {name: sc.name, pass: false, got: 'ERROR ' + (e && e.stack ? e.stack.split('\n').slice(0, 3).join(' <- ') : e), expect: (sc.expect || []).join(' / '), top: ''}; }
			});
			window.PSTest.last = out;
			return out;
		}
	};
})();
