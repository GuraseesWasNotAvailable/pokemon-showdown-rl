# One review pass

This is the routine for improving PSBot while it plays. One pass: look at what has been played
since the last pass, find what cost games, fix it, check the fix, hand the new build to the
running bot, and write down what was done. Run it about once an hour during a long run.

Everything it needs is on disk. Nothing here depends on remembering an earlier pass.

## Where things are

| What | Where |
|---|---|
| The bot's source (edit this one) | `C:\Users\you\ps-ppo\psbot\psbot.js` |
| Recorded games | `C:\Users\you\showdown-bot\battles\<day>\*.json`, index in `battles\index.jsonl` |
| Live status, refreshed every 30 s | `C:\Users\you\showdown-bot\status.json` |
| Bot log, events | `C:\Users\you\showdown-bot\logs\bot-<day>.log`, `logs\events.jsonl` |
| Launcher log | `%LOCALAPPDATA%\PSBot\launcher.log` |
| Deploy ledger, deployed builds | `C:\Users\you\showdown-bot\logs\deploys.jsonl`, `versions\psbot-<build>.js` |
| Review notes (append to it) | `C:\Users\you\showdown-bot\logs\review-log.md` |
| Time of the last pass | `C:\Users\you\showdown-bot\logs\review-state.json` (`{"last": "<ISO time>"}`) |

## The pass

1. **Is it alive?** `python heuristic/report.py C:/Users/you/showdown-bot/battles --since <last>`
   ends with a STATUS line and the recent events.
   - Status older than three minutes while auto-queue was on, `disconnected` true, or `logged in`
     false: read the tail of the launcher log and `events.jsonl` to see what happened. The bot and
     the launcher recover from most of it by themselves, so give it a few minutes before doing
     anything. Being logged out needs the user; note it and stop there.
   - `decide-error`, `rescue` or `rejected` events mean a bug. Find the game and turn in the event
     and fix it first. A bot that errors loses more than a bot that misjudges.
   - Auto-queue off with nothing wrong means the user stopped it. Leave it off.
2. **What was lost?** The report lists the recent losses. For each loss since the last pass:
   `python heuristic/view.py <file>` shows the game a line or two per turn. Find the turn the
   game turned on: a KO that was there and not taken, a Pokemon thrown away, a setup that was
   allowed to happen, a Tera spent for nothing. Then
   `node heuristic/probe.js <file> <turn>` shows every option's score and what the bot believed
   (the opponent's possible moves, the hit it expected, who it thought moves first).
   Many losses are just a worse team or bad luck. Only a move a good player would not have made
   counts as a cause.
3. **Fix causes, not games.** A cause is a rule that is wrong, so the fix is a change to how
   something is scored, not a special case for one Pokemon. One or two causes per pass is plenty.
   If the same thing shows up in several losses, that is the one to fix.
4. **Pin it down.** For each mistake fixed, add the game and the turn as a lesson:
   `node heuristic/lessons.js add <file>` copies the game in with the names removed, then add a
   line to `heuristic/lessons.json` (`want` or `avoid`, matched against the option's label).
5. **Check and ship.** `node heuristic/deploy.js --note "<what changed>"`. It refuses to ship
   unless the file parses, all position tests pass and all lessons pass. It then stamps a build,
   places the file, and waits for the open page to confirm it loaded. The running games carry on
   under the new build. If a change touches how damage or matchups are scored, also run a local
   comparison first when the test server is up:
   `.venv/Scripts/python.exe play.py --agent psbot --opponent simple --games 1000`
   (start the server with `node pokemon-showdown start --no-security` in
   `C:\Users\you\showdown-bot\server`; 90% or better is the bar).
6. **Did the last change help?** Trust the report's BY ARM table over everything else: the two
   arms play the same ladder at the same time, so a gap between them is the change. Leave a change
   in while its arm's interval overlaps the base arm's. Take it out (set it to its base value in
   `PSBot.config`, or start a new `arms.id` with it moved into `base`) when the new arm is
   clearly below, which takes a few hundred games. A bad hour is not evidence: on 2026-10-04 the
   same build went 55% over 127 games and then 28% over 43 with the same decisions, the same luck
   and no fault in the page.
   The report's BY BUILD table gives each build's games, win rate and
   performance rating. A build with 25 or more games that sits clearly below the one before it
   (80 points of performance or more) goes back: `node heuristic/deploy.js --rollback`, and the
   change is reverted in the source. Fewer games than that say nothing either way.
7. **Write it down.** Commit in `C:\Users\you\ps-ppo` on `psbot-hybrid` (local commit, no push).
   Append to `review-log.md`: the time, games and record since the last pass, the rating, what was
   found in which games, what was changed, the build deployed, and anything the user has to do.
   Set `review-state.json` to the time this pass started.

## Limits

- Do not start or stop auto-queue, and do not touch a live battle. `deploy.js` is the only way a
  change reaches the page.
- Never send a chat message, change an account setting, or log in or out.
- Never delete anything. Old builds stay in `versions\`.
- No pushing. Commits stay local until the user asks.
- If nothing clear was found, change nothing and say so in the log.
- If a change cannot be made to pass the checks in this pass, leave the running build alone.
