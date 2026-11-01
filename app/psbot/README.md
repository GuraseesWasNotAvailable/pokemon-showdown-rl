# PSBot

A bot that plays singles random battles inside the Pokemon Showdown client.

It runs in the Showdown page itself. Each turn it reads the same request and battle state the
client uses to draw the move buttons, scores every legal move and switch, and sends the choice
through the client's own send function. It never sends chat messages.

The same file also runs headless under Node (`heuristic/engine.js` in this repo), which is how
the Python side uses it as a teacher and as a guardrail. See `HYBRID.md` in the repo root.

## Files

| Path | What it is |
|---|---|
| `psbot.js` | The bot. Also a valid Tampermonkey userscript. |
| `pstest.js` | 23 position tests rebuilt from real games. |
| `launcher/PSBot.cs`, `build-launcher.cmd` | Source and build script for `PSBot.exe`, the double-click launcher (Windows). |
| `../heuristic/lessons.json`, `../heuristic/lessons/` | The turns that lost recorded ladder games, each with the move the bot should pick there now. |
| `../heuristic/deploy.js` | Runs every check, stamps a build and hands the bot to a running `PSBot.exe`. |

## Loading it

Pick one.

**PSBot.exe (no commands).** Run `build-launcher.cmd` once, then double-click `PSBot.exe`. It
opens Pokemon Showdown in its own Chrome window with the bot already loaded and switched on.
Log in to Showdown in that window the first time; Chrome remembers it after that.

- The window uses its own Chrome profile (`%LOCALAPPDATA%\PSBot\chrome-profile`), because Chrome
  does not allow a program to control your everyday profile. Use it for Showdown only: any program
  on the computer can drive that window while it is open.
- `PSBot.exe` keeps running in the background while the window is open, so the bot comes back
  after a page reload. It exits by itself a few seconds after the window is closed.
- It loads the `psbot.js` sitting next to it, and falls back to the copy built into the exe. When
  that file changes, the new version is loaded into the open page within a few seconds, without a
  reload: the running games carry on under the new version. If the new file will not load, the old
  one is put back.
- It writes next to itself: `battles\` (finished games), `status.json` (what the bot is doing,
  refreshed every half minute), `logs\bot-YYYY-MM-DD.log` (the bot's log) and `logs\events.jsonl`
  (reloads, disconnects, updates, errors).
- While games are running it keeps the computer from going to sleep. See Unattended runs.

**Console (one-off).** Open the Showdown tab, press F12, paste the contents of `psbot.js` into the
Console and press Enter. It lasts until the page reloads.

**Tampermonkey (persistent).** Create a new script, paste `psbot.js` over the template and save.
It then loads on every visit to play.pokemonshowdown.com.

## The panel

A panel appears in the bottom-right corner. Drag it by its title bar; the arrow collapses it.
Its settings are remembered between visits.

| Control | What it does |
|---|---|
| **Start bot / Stop bot** | Whether the bot plays the battles you are in. |
| **Auto-queue** | Finds new ladder games by itself and keeps the chosen number running. |
| **Format** | The random format to queue for. Singles formats only. |
| **Games at once** | How many games to keep running at the same time, 1 to 20. |
| **Games to play** | Stop queueing after this many. 0 keeps going. |
| **Timer** | Turns the battle timer on at the start of every game. |
| **Close finished tabs** | Closes a finished battle tab once the next game is running. The last game stays open. |
| **Skip animations** | Does not wait for battle animations. |
| **Dry run** | Logs the picks without sending them. |
| **Rating** | Your ladder rating after every rated game, the change, the session total, and a graph. Green dots are wins, red are losses. Kept across reloads. |
| **Live battles** | One row per running game: opponent, turn, Pokemon left, last pick. Click a row to open it. |
| **Recorded games** | How many games have been saved. **export** downloads them all as one JSON file. |
| **Log** | What the bot decided and why. |

Two limits come from the public server, not from the bot: it allows about five games at once and
twelve searches every three minutes. The bot notices the server's refusal, holds at that number
and says so in the status line. A local server has no such limits.

If you click a move yourself before the bot does, it leaves that turn alone.

## Unattended runs

Turn **Auto-queue** on and leave it. The bot and `PSBot.exe` between them recover from what
usually ends a long run:

| What happens | What it does |
|---|---|
| The page reloads, or Chrome restarts | Auto-queue picks up where it was, with the same record, if it was on in the last ten minutes. Games that were under way are rejoined and their records continue. |
| The connection drops | Reloads the page after 15 seconds, then backs off (30 s, 1, 2, 5, 10 minutes) if it keeps happening. |
| A search finds nobody for six minutes | Cancels it and searches again. |
| The server says it is restarting or has paused battles | Waits three minutes and tries again. |
| A turn has had no answer for 90 seconds | Sends the server's default choice, so a bug can never time a game out. |
| The page has been open three hours | Lets the running games finish, reloads, carries on (`reloadHours`). |
| The page stops reporting for two and a half minutes | `PSBot.exe` sends it back to Showdown. |
| The Showdown tab is closed while a run is on | `PSBot.exe` opens it again. |
| Chrome crashes while a run is on | `PSBot.exe` starts it again (at most six times an hour). A window you close yourself stays closed. |

To stop a run, turn Auto-queue off, then close the window. A file named `STOP` next to the exe
also keeps it from restarting Chrome.

What it cannot fix: the laptop lid closing (Windows suspends regardless), the power or the network
going away for good, and Showdown logging the account out.

## Ladder experiments
A streak on the ladder looks the same whether a change caused it or not, and a few dozen games
say very little: the same build has gone 55% over 127 games and 28% over the next 43. So a change
to the scoring ships as an experiment. `PSBot.config.arms` names the settings the change added
and what they were before:

```js
arms: {id: 'r3', share: 0.4, base: {freeSetup: 0, phazeTeam: 0, restNap: false, entryRisk: false, revenge: 0, switchRelief: 0, teraSave: false, teraCal: false, swCal: false, keepAnswers: 0}}
```

Each live game is put in an arm by its battle id: `new` plays with the current settings, `base`
with the overrides. The arm is saved in the game's record, and `heuristic/report.py` prints a
BY ARM table with each arm's games, win rate and its 95% interval. Both arms play at the same
time against the same ladder, so the difference between them is the change and nothing else.
About 300 games an arm are needed to see a difference of ten points. `arms: null` turns it off.
Tests, replays and local runs always use the current settings.

## Recorded games

Every game the bot is in is saved when it ends, so it can be gone through afterwards to see
where the bot went wrong. One record holds:

- the format and tier, whether it was rated, both player names, both ratings, the result, and the
  rating before and after
- which build of the bot played it
- your team as dealt: species, level, moves, item, ability, Tera type and stats, and how it stood at the end
- the opposing team as far as it was revealed: species, level, moves seen, item, ability, Tera type, HP and status
- every decision: the turn, both sides' HP at that moment, what the bot chose and why, how long it
  took, and for each option its score, its damage range and accuracy, the chance it takes a KO and
  the chance it costs the Pokemon
- with each decision, what the bot believed: the moves, item, ability and Tera types it thought the
  opponent could have, the hit it expected to take, who it thought moves first, whether it expected
  a switch, a Tera, healing or setup, and what the server offered (PP, disabled moves, trapped, Tera)
- anything that went wrong in the game: a choice the server rejected, an error, a rescued turn
- the turn-by-turn history in plain words, and the complete protocol log (enough to replay the game)

Where they go:

- **With PSBot.exe:** a file per game in `battles\YYYY-MM-DD\<battle id>.json` next to the exe, and
  one summary line per game in `battles\index.jsonl`. Nothing is kept in the browser in that case,
  so a long run does not fill it up.
- **Without it:** in the browser's own storage. **export** in the panel downloads them as one
  file, and `await PSBot.records.all()` returns them in the console.

`PSBot.start({record: false})` turns recording off.

## From the console

```js
PSBot.start()                    // play every battle you are a player in
PSBot.start({autoSearch: true, parallel: 3, maxGames: 30})
PSBot.start({dryRun: true})      // log what it would pick, send nothing
PSBot.stop()
PSBot.explain()                  // score table for the decision on screen
PSBot.log()                      // everything it has decided so far
PSBot.elo.history                // rating after each rated game
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `autoSearch` | `false` | Queue new games by itself. Random formats only. |
| `format` | `gen9randombattle` | Format used by `autoSearch`. |
| `parallel` | `1` | Games to keep running at once, 1 to 20. |
| `maxGames` | `0` | Stop queueing after this many games. 0 means no limit. |
| `autoTimer` | `true` | Turn the battle timer on at the start of every game. |
| `closeFinished` | `true` | Close finished battle tabs once the next game is running. |
| `record` | `true` | Save every game in full (see Recorded games). |
| `fast` | `false` | Skip battle animations instead of waiting for them. |
| `dryRun` | `false` | Decide and log, but do not send. |
| `thinkMs` | `500` | Delay before sending a choice. |
| `switchMargin` | `0.3` | A switch must beat staying in by this much, plus `switchStayTax` (0.6) for every point staying in is still worth. |
| `takeKo` | `true` | A KO that is all but certain beats healing, another boost or a switch. |
| `deepRace` | `true` | Play matchups forward with healing, setup and stat changes. Off gives the plain damage race. |
| `lockPenalty` | `0.15` | Cost of a move that locks you in (Outrage) while they still have a bench. |
| `freeSetup` | `0.5` | Chance a Pokemon with a setup move uses it on the turn you switch, times how sure the move is in its set. 0 scores a switch as if they always attack. |
| `phazeTeam` | `0.35` | Share of what a boosted Pokemon would do to the rest of your team that counts toward Roar, Whirlwind and Haze. |
| `teraCal` | `true` | Chance of the opponent's Tera this turn taken from the recorded ladder games. Off uses the older guess, which ran four times too high. |
| `swCal` | `true` | Raises the weight on the opponent switching where the recorded games showed it low: when you threaten the KO, and when its stats are lowered. |
| `keepAnswers` | `0.5` | Extra cost of HP lost by a Pokemon that is one of the only two able to beat a threat on their bench. 0 turns it off. |
| `restNap` | `true` | Rest in a projection is a full heal and two turns asleep. Off treats it as a free 60% heal, as before 1.4.2. |
| `watchdog` | `true` | The recoveries listed under Unattended runs. |
| `reloadHours` | `3` | Reload the page this often during an auto-queue run. 0 never does. |
| `resumeMinutes` | `10` | How recently auto-queue must have been on for a reloaded page to resume it. |
| `predictSwitch` | `0.12` | Base weight given to the opponent switching out this turn. |
| `switchPressure` | `0.1` | Extra weight on a switch when the opponent's matchup is bad. |
| `sweepWeight` | `0.75` | How much a boost's value against the rest of their team counts. 0 scores setup against the current Pokemon only. |
| `predictTera` | `0.08` | Base weight given to the opponent Terastallizing this turn. 0 ignores it. |
| `teraMargin` | `0.5` | How much better Terastallizing must be to spend it without a KO at stake. `teraEarly` (0.2) raises it while few Pokemon are down, and it drops to 40% once either side is down to two. |
| `teraKoGain` | `0.15` | Gain needed to spend it on securing a KO. Also higher early. |
| `teraSurprise`, `teraStatus` | `false` | The defensive Tera described below. Off by default: it spent Tera too freely in real games. |

## How it decides

Every option gets a score in units of material: the HP fraction the opponent is expected to lose
minus the HP fraction you are expected to lose, plus a bonus for taking a KO.

- **Damage.** A full damage calculation per move: stats, boosts, STAB and Tera STAB, type chart for
  the generation, weather, terrain, screens, items and around 80 abilities, plus special cases such
  as Body Press, Psyshock, Foul Play, Gyro Ball, multi-hit moves and Knock Off.
- **Your team.** Stats, moves, abilities and items come straight from the server.
- **The opposing team.** Level and species are known. Stats are estimated by solving for the EV
  spread from your own team, since random formats give every Pokemon the same spread. Moves,
  abilities, items and Tera types use what has been revealed, then the possible sets for that
  species from the public random-battle set data.
- **Turn order.** Speed, priority, paralysis, Choice Scarf, weather abilities, Tailwind and Trick Room.
- **The exchange.** For each move it works out the chance you KO first, the chance you are KO'd
  before or after moving, and how the one-on-one plays out from the resulting position.
- **Playing the matchup forward.** The one-on-one is played out turn by turn on expected damage.
  Either side heals when that keeps it alive and gains on what comes in, so a Pokemon that
  Recovers more than you deal is not something you "win the race" against: the bot stops clicking
  the same attack into it and looks for a boost, poison, hazards or a stronger teammate. The
  opponent may open with one or two turns of a setup move it has shown or probably carries, and
  the line that is worst for you counts, weighted by how sure the move is. Stat changes from the
  moves themselves are applied as they happen: Close Combat's defence drops, Draco Meteor, a Speed
  boost from Aqua Step that changes who moves first. Priority moves finish a weakened Pokemon.
  Substitute, Disguise, Ice Face, Sturdy and Focus Sash take the hit they take, on either side and
  once. Life Orb and recoil are paid for, including when they take the last HP. Rest is a full heal
  followed by two turns asleep (none with a Chesto Berry or Sleep Talk), so a Pokemon with Rest in
  its set can still be worn down.
- **Take the KO.** A KO that is all but certain and safe comes before healing, another boost or a
  switch, however good the boost looks. The exception is setting up on something that cannot hurt
  you. Between two KOs that score about the same, it takes the one more certain to land.
- **Switching.** A switch is scored after the incoming Pokemon takes the hit, hazard damage, poison
  from Toxic Spikes and the Speed drop from Sticky Web, and with its Intimidate applied. The better
  staying in still is, the more a switch has to gain, and the worse it is, the less. A Pokemon
  that is asleep with no way to act, or about to go down without a KO, leaves for any clearly
  better teammate. It will not bounce straight back to a Pokemon that just left.
- **Coming in.** The hit on the way in is either the move they picked for the Pokemon that is
  leaving or, if they read the switch, their best move against the one arriving. If that hit can
  kill, the switch is scored as a death with that chance, not as an average the Pokemon appears to
  survive. The same goes for the teammate that comes in after a U-turn that moved first. And when
  the Pokemon on the field is about to go down anyway, the free switch that follows the faint is
  counted against switching now, so a healthy teammate is not thrown into the hit to save one that
  is already lost. A Pokemon that would go down to the hazards on the way in is never the one sent.
- **The free turn.** The turn you switch is a free turn for them. In the recorded ladder games a
  Pokemon with a setup move used it on about half of those turns, against a quarter of the turns
  the bot stayed in. So the Pokemon coming in is also scored against the boosted version: with that
  chance it takes no hit on the way in and faces the opponent a stage or two up. That favours a
  teammate that still wins the matchup after the boost, and staying in to hit when nobody does.
- **The opponent switching.** It names the benched Pokemon the opponent would most likely bring
  in (the one that takes your obvious move best) and weighs every option against it as well as the
  Pokemon in front. The weight follows what this opponent has actually done: it is small until
  they have shown they switch, and never above one half, so a move that works against what is
  really on the field is not given up for a guess. Two corrections come from 5573 recorded turns:
  with a KO threatened they left 7 points more often than weighted, and with lowered stats about
  15 more.
- **Setting up.** A boosting move is scored by what the boost does to the rest of the opposing
  team, not only to the Pokemon in front: how many of them you then beat that you would not have.
  That is what makes Swords Dance, Dragon Dance, Calm Mind, Shell Smash and the rest worth a turn.
  It sets up when the turn is cheap (the opponent is asleep, frozen, passive, or cannot hurt you
  much) and attacks when it is not. A boost is worth nothing in front of a Pokemon that has, or
  probably has, Haze, a phazing move, Encore or Unaware. It plans for the hit the opponent most
  likely has rather than the average, and for what is left after the move's own cost, so it does
  not Belly Drum into a hit that finishes it. White Herb is accounted for.
- **Lowering their stats.** An attack that drops the target's stats (Mystical Fire, Lunge, Spirit
  Break, Rock Tomb and so on) is scored by how the rest of the fight goes after the drop. Status
  moves that lower stats are scored the same way, and avoided against Contrary, Defiant and
  Competitive.
- **When they set up.** A boosted Pokemon is worth more to knock out than its HP says, and while
  the opponent is boosting, a turn that does nothing about it costs something. Roar, Whirlwind and
  Haze are scored by what the boosts would have done to the rest of your team, not only to the
  Pokemon using the move, as long as it survives the hit to use it.
- **The opponent's Tera.** A Pokemon that has Terastallized is treated as its new type, for the
  hits it takes and for its STAB. While the opponent still has its Tera, the bot looks up the Tera
  types that Pokemon's sets allow and gives a little weight to it using one this turn. The weight
  comes from what opponents did over 4276 recorded turns: 3.8% a turn overall, from 0.6% with
  nothing at stake and a full team, up to about 20% when a Tera would save the Pokemon from a KO
  and they have lost three or more. A boosted Pokemon uses it more, and almost nobody does in the
  first two turns.
- **Keeping the answer.** In three recorded losses out of four, one opposing Pokemon took three
  or more KOs. The bot was ahead when that started in 63% of them, and had seen the Pokemon at
  least three turns earlier in 56%. So it looks at what is waiting on their bench. If a Pokemon
  there beats two or more of yours and only one or two of yours beat it, the HP of those one or
  two costs more than its share, whether it is lost staying in or on the way in.
- **Your Tera.** Held for when the game has taken shape, with one more reason to spend it: when it
  turns a hit that would take the Pokemon out this turn into one it survives. In 500 games between players rated
  2400 and up, Tera was spent on turn 19 on average with two of the player's own Pokemon already
  down, and by turn 2 in only 2% of games; the winners took a KO on their Tera turn 41% of the
  time. So the bar starts high, comes down as Pokemon fall on either side, and is lowest when it
  secures a KO. It is not spent while asleep, into a Substitute, or when the opponent is likely
  to switch out of the hit. When it is spent and one of the Tera moves is a sure KO, that is the
  move. A defensive Tera on a status-move turn is available behind
  `teraSurprise` and `teraStatus`.
- **Sucker Punch.** Worth the chance the opponent attacks. That chance is lower against a Pokemon
  with a setup or recovery move and reason to use it, and it halves every time Sucker Punch fails
  against the same Pokemon.
- **Locking moves.** Outrage and the like cost something while the opponent has a bench, and more
  when a revealed teammate is immune. A Choice item locks too: after a KO the same move has to
  deal with whatever comes next, which counts in full when there is nobody left to switch to.
- **What it has not seen.** An unrevealed item or ability is averaged over the sets the Pokemon
  can run, so a Choice Band or Tough Claws that is probably there is in the number. One move since
  it came in fits a Choice item; two rule it out. A random-battle Pokemon that has shown Trick Room
  or Gyro Ball was built with no Speed, and is treated that way. Checked against 2300 recorded
  hits, predicted damage is within 1% of actual on average. The exceptions it turned up are now
  handled: Regigigas under Slow Start, the four Ruin abilities, a charged Electric move, and
  Morpeko's Aura Wheel.
- **Leaving a lost matchup.** A Pokemon that is about to go down without taking anything with it
  switches out for any clearly better teammate, instead of waiting for the usual margin.
- **Status moves.** Setup, recovery, hazards, status, screens, Encore, Taunt, Haze, Protect and
  pivots each have their own rule. It will not set up in front of something that hits hard or heal
  a healthy Pokemon.
- **Mega, Dynamax, Z-Moves.** Mega Evolution is always used. The others are spent when they
  improve the position by a clear margin, or late in the game.

## Tests

```
node heuristic/setup.js --all-gens     # once: fetches the Showdown client files and the move data
node heuristic/test.js                 # the 23 position tests, headless
node heuristic/lessons.js              # the turns that lost ladder games, replayed through the current bot
node heuristic/replay.js <battles folder> --losses   # what the current bot would do in recorded games
node heuristic/probe.js <game> <turn>  # every option's score and what the bot believed on one turn
python heuristic/view.py <game.json>   # one recorded game, a line or two per turn
python heuristic/report.py <battles folder>          # record, rating, per-build results, live status
node heuristic/toplay.js fetch && node heuristic/toplay.js stats --mine <battles folder>
                                       # how the best players' habits compare with the bot's
node heuristic/deploy.js --note "..."  # all checks, then hand the build to a running PSBot.exe
node heuristic/deploy.js --rollback    # put the previous build back
```

A lesson is one line in `heuristic/lessons.json`: a recorded game, a turn, and what to pick or
avoid there. `node heuristic/lessons.js add <game.json>` copies a game into `heuristic/lessons/`
with the players' names removed. `deploy.js` refuses to ship a build that fails any of them.

In a Showdown page, load `pstest.js` after the bot and run `await PSTest.run()`.

## A note on the move data

The Showdown client replaces each entry of its move table with a cut-down object the first time
it looks that move up, which it does for every move used in a battle and for the buttons of the
Pokemon on the field. The cut-down object has no boosts, drain, pivot, hazard or stat-override
fields. The bot therefore keeps its own copy: whatever is still intact when it loads, completed
from the site's `data/moves.json`. Before 1.4.0 it read the client's table directly and could
mistake a move it met late (its own lead's moves, in the first games after a page load) for one
that does nothing special.

## Limits

- Singles only. Doubles and multi battles fall back to the server's default choice.
- It does not search the game tree. It reasons about this turn and the current matchup.
- It cannot see through Zoroark's Illusion.
- Formats with your own team work if you start the battle yourself, but auto-queue only queues
  random formats.

## The public server

On play.pokemonshowdown.com, ladder opponents are other people's accounts. Check Showdown's rules
on bots before running this on the ladder at volume.
