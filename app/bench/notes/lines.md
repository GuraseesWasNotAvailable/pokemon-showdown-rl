# Sol decides, and reading ahead (2026-10-13)

Three things asked for together: a choice between Luna and Sol as the model that picks the move, the
same in the window app, and a search that follows its candidate moves 4 to 5 turns and compares
where they end. This note says what each is, what was measured, and what was not.

| Arm (`--player`, `--arms`) | What plays |
|---|---|
| `mix` | today's bot: the network and the search, the veto rule (unchanged) |
| `luna` | mix, then OpenAI's gpt-6-luna picks among the legal actions (unchanged) |
| `sol` | mix, then OpenAI's gpt-6-sol picks (new) |
| `mix-lines` | mix, then the lines: mix's pick gives way to an action whose line ends clearly better (new) |
| `luna-lines`, `sol-lines` | mix, the lines, then the model picks with the lines in its brief (new) |

The window app's list "Who decides" and its box "Read ahead" choose among exactly these;
`mix-deep` (the pre-registered A/B's arm: 5 times the search, no lines) is not touched.

## 1. Sol

`luna_player.MODELS["sol"]`: model `gpt-6-sol`, `POST https://api.openai.com/v1/responses`. The
Decisions API takes gpt-6-luna only, so Sol is asked the same question in the Responses API's
shape (`responses_body`): the instructions as the system message, the brief and the choices as the
user's, and a strict JSON schema `{"action": {"reason", "choice" (an enum of the offered values),
"confidence"}}`; `responses_answers` reads the reply back into the Decisions shape, and from there
on Luna's code is Sol's: the rails, the blunder guard, the counts, the records.

* Reasoning effort `none` (`--sol-effort`): no hidden reasoning, one short "reason" in the answer,
  which the console and the window show when Sol's pick is not mix's. OpenAI's model page lists
  `none` for gpt-6-sol and its reasoning guide reads as if it did not; so when the API answers
  HTTP 400 naming the effort, `low` is used for the rest of the run, said once (tested against the
  fake). With reasoning the answers take longer and cost more (reasoning tokens are output tokens).
* Time. A model's arm with a budget gets seconds of its own after the search
  (`SearchBudget.decide_s`): Sol 5 s (`--sol-decide-s`, `--sol-timeout`), so a decision is the
  search (within 2.5 s) and then Sol, 7.5 s at the very worst and about 3 s as a rule. The Luna arm
  is as it was: Luna gets what the 2.5 s leave. Under 90 s on the battle timer the arm plays
  today's budget, in which Sol is asked only if time is left (it rarely is): mix plays.
* Money. $2 per 1M input tokens, $10 per 1M output: about half a cent a decision (the check prints
  it from the API's own token counts), 15 to 20 cents a game. Each decision's cost is in its record
  (`luna.cost_usd`) and each game's in results.jsonl (`luna_cost_usd`, `luna_model`). `--max-model-usd`
  (20) turns the models off for the rest of a run once its decisions have cost that much.
* The key is the one key for both models: `OPENAI_API_KEY`, the hidden prompt, or, for the window
  app, the line after the password's on standard input (`--key-stdin`).

## 2. The lines

`bench/fp_bridge_server.py` "lines", asked with the search (`SearchBudget.lines_ms`): after the
ordinary search of every sampled world, for the 2 likeliest worlds and our 3 actions the search
favours most, a line is played: we play the action and the opponent its most visited reply; the
engine's likeliest outcome of the pair is applied; the new position is searched for about 36 ms on
4 of the engine's threads; both sides play what that search visited most; and so on for 5 whole
turns (a forced switch is a step, not a turn) or until a side has nobody left. The line's value is
the search's mean result at its last position (0 lost, 1 won), with the first step's other
outcomes of 5% or more (a miss, a critical hit) each searched once and weighed in by their chance.
1.5 s in all; the search and the lines have 4.6 s.

What the lines are not: the search itself already looks ahead, and its share and result per action
come from hundreds of thousands of lines. One greedy line per action is a different and much
noisier reading of the same engine. What it adds is a concrete future per action, in words.

Measured on the real engine (the box, recorded battles replayed through the helper, 500 ms search
then the lines; `rule_rate.py`, 66 decisions that had a line for the search's favourite and another):

| | 1 thread | 4 threads |
|---|---|---|
| The lines' time, mean (slowest) | 1,054 ms (1,543) | 1,350 ms (1,908) |
| Asked to answered, mean (slowest) | 2,147 ms (2,667) | 2,444 ms (3,050) |
| The same action's line in the two worlds differs by, mean (median) | 0.121 (0.081) | 0.126 (0.093) |
| Another line beats the favourite's by 0.08 | 27% of decisions | 15% |
| ... by 0.10 and is worse in neither world (the rule) | 21% | 8% |
| The favourite's line minus the search's own result for it | -0.036 | -0.037 |

So one world's line against another's differs by about 0.12 for the same action: a margin of 0.08
on the mean alone fires on noise. Hence the rule of `mix-lines` (`search_player.LinesRule`,
`lines_pick`): another action is played only when its line ends 0.10 or more above the line of
mix's pick, the search itself rates that action (a quarter of its largest share), and in no world
read does its line end below mix's. With one thread the step's search is thin and the lines
contradict the search more often; with four they agree more, which is why four is the default.
A line in the first battle looked at had Dragonite use Dragon Dance twice into an Avalugg's
Avalanche and faint: 36 ms on one thread does that. Read the lines as a rough second opinion.

## 3. Tested

All on the cloud box. No key of OpenAI's anywhere, and no call to OpenAI.

* `tests/test_sol_lines.py` (35 tests): the Responses request held to the documented shape and to
  the strict-schema rules by the fake (`tests/fake_decisions.py`, now also `/v1/responses`); the
  answer's shapes (a choice, a refusal, incomplete, not JSON, a value not offered); every API
  failure; the effort fallback; the prices; the check for either model; the arms, flags and
  budgets; the key on standard input; `ladder.main` reading the password's line and then the
  key's and telling the app why it stopped; the lines on our 14 actions; the rule; the brief's
  LOOK-AHEAD section word for word; the arms in the player on recorded battles (Sol answering
  mix's pick plays order for order what mix plays; Sol's seconds after the search; a model's
  brief with and without lines; `mix-lines` changing the move exactly where the rule says); the
  spending cap; the launcher; the window app's source; and the real helper playing a whole
  recorded battle's lines (82 lines, 74 of five turns, 13 ended the game, 653 ms on average).
* The suites the change touches pass as before: test_luna 88, test_search_player 55,
  test_ladder_ab 53, test_foulplay_bridge 96, test_foulplay_remote 29, test_ladder_login 28,
  test_counter 79, test_setup_fix 18, test_predict_live 12. test_obs_abil and test_own_tera fail
  as they did before this work (they test training code this branch does not have).
* The window app: built with `build-app.cmd`'s compiler, opened against a test address in each
  mode, and its window read from a picture (the list, the box, the key field, what is greyed out).

### Bot against bot: mix and mix-lines against the same opponents (bench/lines_gauntlet/)

254 games on the box in 42 minutes (`run.sh`: the gauntlet's own scripts, the same player playing
both arms, the arm balanced within each side, network 024381, 5 games at a time on 6 helpers).

| Opponent | mix | mix-lines | mix-lines minus mix |
|---|---|---|---|
| Foul Play at fair compute (450 ms, its own helpers) | 29-28 (50.9%) | 37-20 (64.9%) | +14.0 points (95% -4.0 to +30.8), z 1.5 |
| The network alone (024381) | 38-32 (54.3%) | 29-41 (41.4%) | -12.9 points (95% -28.4 to +3.6), z -1.5 |
| Both | 67-60 (52.8%) | 66-61 (52.0%) | -0.8 points |

Level, and too few games to say more: the two opponents point in opposite directions, each inside
its own noise. It is a check that reading ahead does not break the bot, and it does not: no late
decision in either arm, no invalid order, no helper timeout or exit in 9,291 decisions (7
"unavailable choice" answers in the phase against the network, the server's way of showing a
trapped or disabled move, not told apart by arm). A decision of mix-lines took 2.6 s on average,
3.1 s at the 95th, 3.3 s at the slowest (mix: 1.15, 1.23, 1.38). The lines were read on 4,946
decisions and changed the move on 373 (7.5%): against Foul Play on 6.7% of the decisions of won
games and 7.3% of lost ones, against the network on 6.4% and 9.1%. The lines play Terastallization
more: 12 and 14 of the Tera turns were moves the lines changed (Tera in 93% and 91% of mix-lines'
games, 81% and 89% of mix's).

### Real games of the model arms with a fake API (bench/lines_gauntlet/model_smoke.sh)

10 games on a local server against the network alone, `--arms sol-lines,luna-lines`, ladder.py run
as the window app runs it (`--app-events --password-stdin --key-stdin`, the key the second line of
standard input), the models played by `tests/fake_decisions.py` answering 0.7 to 1.8 s late with 8%
of its answers errors.

| | sol-lines | luna-lines |
|---|---|---|
| Decisions | 150 | 133 |
| A decision took, mean (95th, slowest) | 3.85 s (4.72, 6.03) | 2.74 s (4.09, 4.27) |
| Late decisions, decisions over 9.5 s | 0, 0 | 0, 0 |
| The model asked, answered | 147, 138 (1.3 s on average) | 44, 17 |
| Mix played instead | 9: HTTP 429 4, HTTP 500 1, malformed 2, illegal 1, refusal 1 | 110: paused 83, timeout 24, others 3 |
| Counted cost (the fake's token counts at the real prices) | $0.65 (0.44 cents a decision) | $0.009 |

Both checks ran before the first game, each model through its own endpoint. Luna's column is the
rails at work, not Luna: the fake's delay was set for Sol, so most of its answers came after Luna's
1.2 s and the breaker paused it (the real Luna answers in about 0.15 s). The app's events came out
whole: 1 start, 10 game starts and ends, 283 moves with whose move each was, 1 finished.

## 4. Not tested, because it cannot be here

* The real API for Sol: whether it takes effort `none`, how long an answer of a game's size takes
  from this laptop, what a decision really costs. The check before the first game prints all
  three and stops the run when Sol cannot answer in its 5 s.
* Whether Sol plays better than Luna, or either better than mix. In the 50 ladder games of
  2026-10-12 Luna agreed with mix on 95% of its answers and changed 78 moves, 61 of them in the 23
  losses. `--arms mix,sol` (Luna Play asks) measures Sol the same way.
* Whether reading ahead wins more on the ladder. The bot-against-bot games above are a check that
  it does not break the bot, not a verdict; a verdict needs about 250 ladder games per arm.
* The window's Start with a real password and a real key (they are typed by the account's owner).
  Both ends of that pipe are tested: ladder.py with the two lines, and the app's source for what
  it writes.
