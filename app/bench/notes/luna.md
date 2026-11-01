# Luna decides: OpenAI's Decisions API picks the move

2026-10-12, branch `luna` (from `ladder-ab` at bd9fb4b). Files: `luna_player.py` (new),
`luna_check_sample.json` (new), `search_player.py` and `ladder.py` (the arm `luna`),
`bench/ab_report.py` (the Luna line), `bench/luna_report.py`, `bench/luna_audit.py`,
`bench/luna_sample.py`, `bench/luna_fixture.py` (new), `tests/test_luna.py`, `tests/fake_decisions.py`,
`tests/fixtures/luna_ladder_games.jsonl.gz` (new), `tests/test_search_player.py` (the list of modes),
`launchers/laptop/Luna Play.cmd` and `Luna Stop.cmd` (new).

Asked for: "a pokemon showdown model using OpenAI luna decision models which I just need to give an
api key to and that will give you the decision on what to do and it executes that automatically".

    Luna Play.cmd                                      the launcher (Desktop)
    python ladder.py --luna-check                      the key, the endpoint, the answer's shape; nothing is played
    python ladder.py --arms mix,luna --experiment luna-vs-mix-024381 ...     the A/B the launcher runs by default
    python ladder.py --player luna ...                 Luna in every game
    python bench/luna_report.py                        what Luna did in its games so far
    python bench/luna_sample.py                        the brief and the question of a real logged decision

**Nobody can promise a rating.** Today's bot (mix on 024381) is about 2,350. Luna gets everything
mix knows and is told to follow it unless the facts give a reason not to, so it should not fall far
below mix, and it may do better where mix is known to be weak (a sure knockout passed over for a
90% move, a boosted sweeper stayed in on). Whether it does is exactly what the A/B measures, and
30 games cannot say: with 15 games an arm the difference has a standard error of about 18 points.
`bench/ab_report.py` gives its verdict at 250 games an arm (interim) and 500 (final), as for the
other A/Bs, and until then says "for information only".

## 1. The API, as its guide documents it

https://developers.openai.com/api/docs/guides/decisions (read 2026-10-12; public beta, GA "in the
coming weeks"; `gpt-6-luna` is the only model).

    POST https://api.openai.com/v1/decisions      Authorization: Bearer KEY      Content-Type: application/json
    {"model": "gpt-6-luna", "input": "<text>" (or user messages of input_text / input_image parts),
     "questions": [{"type": "choice", "name": "...", "instructions": "...",
                    "choices": [{"value": "...", "description": "..."}]}]}       (also "predicate" and "score")
    -> {"answers": [{"type": "choice", "name": "...", "choice": "<value>",
                     "probabilities": [{"value": "...", "probability": 0.95}, ...], "confidence": 0.93}]}
       a predicate: {"type": "predicate", "name", "probability"}; an answer can be {"type": "refusal", "name"}

Input costs $0.10 per 1M tokens and nothing else is charged. What the guide does NOT say, and the
code therefore does not rely on: there is no `usage` in the answer (the tokens are estimated here,
and read from `usage` should the API send one), no error shape (OpenAI's usual
`{"error": {"message", "type", "code"}}` is read when it is there, else the start of the body), no
limits on the number of choices or the length of a description, no rate limits and no latency
figure ("about 10x faster than the Responses API"). The check below asks a question of a game's
size for that reason. Plain HTTPS with `requests`; no SDK.

## 2. What happens at a decision

In a `luna` game (the arm is "mix with one more step": `split_mode("luna") == ("mix", None)`):

1. Everything mix does today, unchanged: the network's forward pass, Foul Play's search (500 ms,
   answer within the 2.5 s deadline), the read of the opponent, mix's own pick.
2. With more than one legal action: the brief and the question are written
   (`luna_player.build_request`, 1.7 ms on average), and Luna is asked ONE choice question whose
   choices are exactly the legal actions.
3. Luna's choice is played, unless a rail says otherwise (section 4).

The brief (`input`): the turn and the kind of request (a move turn, a forced switch after a faint or
a pivot), how many Pokemon each side has left, the network's win estimate; our active Pokemon
(level, typing, HP, status, boosts, item, ability, Tera type); the opponent's active Pokemon (the
same, its item and ability as shown, else marked as a guess: "likely", "possible", "its sets all
have"; the moves it has used; the moves it may also have; whether it can still Terastallize); the
speed order with its doubt (a tie within 4%, a Choice Scarf in its sets, a priority move it has
shown, Trick Room); the threats (its boosts, a set-up move shown or likely, the read that it sets up
or puts us to sleep, how hard its seen and its possible moves hit our active, our answer on the
field and which answers cannot work now); the opponent's bench as seen; the field, weather,
hazards and screens; what happened last turn (or this turn so far, at a forced switch); the
opponent's likely next action from `opponent_model.py`; and what mix, the network alone and the
search alone would play.

Each choice (`value` "a<our action id>-<name>", `description`): a move with its type, category,
power, accuracy, priority, what else it does, the estimated damage range on the current target
(`setup_counter.damage_share` at the lowest and highest roll, screens counted) with the KO chance
from the opponent's HP and the accuracy, what could stop it (an immunity ability it may have, a
Focus Sash or Sturdy at full HP), and for a Tera form what the new typing does to the hits we
take; a switch with the Pokemon's level, typing, HP, status, item, ability and moves, how hard the
opponent's seen and possible moves hit it, its best hit back, whether it is faster, and what the
hazards on our side do to it. Then `net N%` (the network's probability, a move and its Tera form
together), `search N% (result 0.xx)` (the search's share of exactly that action and its mean
result), and `MIX'S PICK` on the one mix would play.

The question's instructions (`luna_player.INSTRUCTIONS`): pick the action most likely to win the
game; what `net` and `search` mean and that together they play at about 2,350, to be followed
unless the facts give a concrete reason not to; respect immunities and abilities; do not stay in on
a boosted sweeper without an answer; take a safe knockout; convert a lead safely; do not waste
Tera; do not switch without a reason.

Nothing hidden and nothing a person typed: the brief is built from poke-env's battle (what the
room sent our side), Foul Play's answer (its samples of the opponent's hidden sets are guesses from
the public random battle sets, and are written as guesses) and the public dex. No user names, no
chat (`luna_player.TurnLog` keeps only what the Pokemon did).

A sample (a real logged decision, turn 10 of a ladder game; `luna_check_sample.json` is this
request): mix plays Draco Meteor (the network 96%), while the facts say Earthquake is a sure
knockout ("accuracy 100% ... KO chance about 100%") where Draco Meteor is "KO chance about 90%"
and costs two stages of Special Attack. That is the kind of decision Luna is there for.
`python bench/luna_sample.py --game 1 --turn 10` prints it.

## 3. The key

`OPENAI_API_KEY` when it is set, else typed once at a hidden prompt (`getpass`; the launcher asks
with PowerShell's `Read-Host -AsSecureString` so that the check and the run share one prompt). It
is held in the process's memory and put into the `Authorization` header, nowhere else: there is no
flag for it, `ladder.py` takes the variable out of its own environment before it starts anything
(the search helpers do not inherit it), and the launcher clears it when the run ends. Everything
printed about a failure goes through `luna_player.redact` (the key, anything shaped like a key
such as the API's own half-masked echo, a Bearer token, an Authorization header: `***`), and the
console log masks the key in the file and in the window as well. The key is sent to
`https://api.openai.com` or to a test server on 127.0.0.1 only: any other `--luna-url` is refused
before a request is made.

## 4. The rails (each counted per game and written in the decision's note)

| what | then |
|---|---|
| under 0.25 s left before the decision's deadline (`no_time`) | not asked, mix's pick |
| no answer within `--luna-timeout` 1.2 s, or what the 2.5 s deadline leaves if that is less (`timeout`) | mix's pick |
| HTTP error (`http_429`, `http_500` ...; a 5xx and a dropped connection are tried once more when 0.3 s are left, a 4xx and a 429 are not) | mix's pick |
| an answer of type `refusal` | mix's pick |
| an answer that is not the documented shape (`malformed`) | mix's pick |
| a choice that is not one of the values offered (`illegal`) | mix's pick |
| 5 API failures in a row (timeouts, HTTP errors, wrong shapes; not refusals or illegal choices) | not asked for 60 s (`paused`), one console line |
| the API refuses the key or the model (401, 403, 404) | Luna is off for the run (`key_refused`), one console line, no new games are started, the games in progress are finished by mix, exit code 5 |
| the blunder guard (on; `--luna-guard off`): Luna's pick under 1% of the network AND under 1% of the search, mix's over 60% of either | mix's pick, the note says `overruled` |

A decision of a Luna game is never later than a mix one's deadline: the wait for Luna is cut at
the decision's own deadline.

## 5. The records

Decision note (`runs/ladder/battles/<battle>.jsonl`, `{"decision": {..., "luna": {...}}}`):
`asked`, `mix` (what mix would play), `choice` (our action id) with `choice_text` and
`choice_value`, `confidence`, `probabilities` (action id to probability), `latency_ms`,
`input_tokens` with `tokens_from` ("estimate", or "usage" when the API sends one), `fell_back`
and `why`, `overruled` and `overruled_why`, `timeout_ms`, `brief_ms`, and (unless
`--no-luna-save-briefs`) `brief` and `choices` as they were sent. The record's `source` is "luna"
when Luna's choice was the order.

Results line (`runs/ladder/results.jsonl`): `luna_decisions` (decisions with a choice),
`luna_asked`, `luna_answered`, `luna_played`, `luna_overruled`, `luna_agreed_with_mix`,
`luna_changed`, `luna_fallbacks` (by reason), `luna_latency_ms_mean`, `luna_latency_ms_max`,
`luna_tokens`, `luna_cost_usd`. The game's console line, the "by arm" lines and the end-of-run
summary print them, `bench/ab_report.py` adds a Luna line per arm and warns when Luna's choice was
the order on under 80% of the arm's decisions (then the arm is mostly mix), and
`bench/luna_report.py` adds them up over all games.

## 6. Tested (all on the cloud box; no OpenAI key anywhere, no call to OpenAI)

`tests/fake_decisions.py` is a fake of the API on 127.0.0.1 that holds every request to the
documented shape and answers as told. `tests/test_luna.py` (82 tests; the one against a real
tokenizer needs tiktoken, and was run with it once):

* the request on the wire is exactly the documented one; a normal answer is played;
* a slow answer, HTTP 400/401/403/404/429/500/503, a web page, a dropped connection, a refusal, three
  kinds of wrong shape and a choice that was not offered: each plays mix's pick and is counted by
  its reason, at the client, at `Luna.choose`, and in the player on recorded battles;
* the breaker (5 failures, 60 s), the key refused (off for the run), the blunder guard on and off;
* the key: never in an error's text, a record, a results line, the console (window and file), or
  `repr`; a wrong key's echo by the API comes out as `***`; never sent to another host;
* identity: with Luna answering mix's pick, a `luna` player sends exactly the orders a `mix` player
  sends on the seven recorded battles (156 decisions with a choice); with any failure, the same;
  `net`, `mix` and `search` players never touch Luna;
* the brief on 347 real decisions of ten logged ladder games: at most 2,028 tokens by the estimate
  (mean 1,427; budget 2,500), every part written, the choices exactly the legal actions, mix's pick
  marked once, and nothing hidden: no opponent Pokemon named before it came out, no move listed as
  seen before it was used, no item or ability stated before the battle showed it, no user name;
* `--luna-check` (and the check a run starts with): what it prints, the exit codes, one line on
  failure; the launcher's lines parse and play what they say.

The estimate of the tokens is 1.04 to 1.11 times the count of the `o200k_base` encoding on these
requests (measured with tiktoken on the box; which tokenizer gpt-6-luna uses is not published).
By that encoding a question is about 1,330 tokens (1,921 at the most), so about $0.00013 a
decision and about half a cent a game of 40 decisions.

The whole suite on this branch: 809 passed, and the same 35 tests fail as on `ladder-ab` at bd9fb4b
(728 passed there): tests of `test_obs_abil`, `test_own_tera` and `test_speed_kvpool`, which need
files of other branches. (Two runs of the suite at once fail `test_pipeline`: its fixed ports.)

Smoke games (`ladder.py --arms mix,luna` on a local Showdown server against SimpleHeuristics, the
real search helpers, 3 games at a time, the fake API), each checked from its files by
`bench/luna_audit.py` (see section 7 for the numbers of the last runs):

* (a) the fake always answers mix's pick: 24 games;
* (b) random legal picks, a random delay of 0.05 to 0.9 s, 20% errors of nine kinds: 24 games;
* (c) the key stops working after 25 questions (HTTP 401): Luna off, no new games, exit code 5;
* (d) HTTP 500 after 12 questions: 5 failures, the pause of 60 s.

## 7. The smoke runs' numbers

Box, 2026-10-12, `~/dev/luna-out/luna-smoke-a4` and `-b4` (the final code) and `-c3`, `-d3`, `-p3`
(the code before a last change of the instructions' wording), each with `report.txt`: the A/B
report, Luna's report, the audit. Checkpoint 024381 on the CPU, 500 ms of search on 4 helpers,
deadline 2.5 s, 3 games at a time.

| run | games (mix, luna) | Luna asked | answered | played | overruled | mix instead (why) | request to order, luna games | late | illegal or refused orders | briefs with something hidden | key in files |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) mix's pick | 24 (12, 12) | 243 of 243 | 243 | 243 | 0 | 0 | 1,142 ms mean, 1,267 slowest | 0 | 0 | 0 of 243 | 0 |
| (b) random, delays, 20% errors | 24 (12, 12) | 430 of 430 | 352 | 325 | 27 | 78 (malformed 24, timeout 21, http_429 14, refusal 9, illegal 8, http_500 1, http_503 1) | 1,633 ms mean, 2,413 slowest | 0 | 0 | 0 of 430 | 0 |
| (c) HTTP 401 after 25 questions | 4 of 12 started (2, 2) | 26 of 50 | 25 | 25 | 0 | 25 (http_401 1, key_refused 24) | | 0 | 0 | 0 of 26 | 0 |
| (d) HTTP 500 after 12 questions | 6 (3, 3) | 17 of 63 | 12 | 11 | 1 | 51 (http_500 5, paused 46) | | 0 | 0 | 0 of 17 | 0 |
| (p) `--player luna`, never mix's pick | 6 (0, 6) | 125 of 125 | 125 | 117 | 8 | 0 | | 0 | 0 | 0 of 125 | 0 |

In every run the counts of the results lines equal the counts of the decision notes, no game was
lost on time, and no decision raised. Mix's own decisions in the same runs took 1,129 and 1,137 ms
on average (1,220 and 1,329 at the slowest): in (a) Luna's step added about 13 ms with a fake that
answers at once. In (b) the fake's answers took 533 ms on average (95th 1,178, slowest 1,208 ms).
Tokens per question by the estimate: mean 1,276 to 1,568, largest 2,013 (budget 2,500).

(c) printed, once each: "Luna is OFF for the rest of this run: the API refused the key or the model
(HTTP 401: Incorrect API key provided: ***. ...)", "STOPPING: Luna is off (...). No new games,
finishing the 3 in progress with mix's moves.", and at the end "stopped early: Luna was turned off
in the middle of the run (...); 4 of up to 12 games were started", exit code 5. (d) printed "Luna:
5 API failures in a row (the last: HTTP 500: ...). Not asked for 60 s: today's bot (mix) plays
meanwhile." and went on (exit code 0).

That Luna's choice is the order shows in the results: with random picks the luna arm won 1, 3, 3
and 1 of 12 games in the four rounds of (b), where mix won 12, 12, 10 and 11 (z about -11 in the
first), and with "never mix's pick" it won 0 of 6. With mix's pick it won 11, 12, 11 and 12 of 12
(mix: 10, 12, 10, 11).

## 8. Not tested, because it cannot be without a key

* The real API: that the answer is the documented shape (the check verifies it before any game),
  how long an answer of a game's size takes from this laptop (the check measures it and warns when
  it is over the 1.2 s), its rate limits (3 games at a time ask about 2 questions a second at the
  most; a 429 plays mix's move and five in a row pause Luna for a minute), and whether it refuses
  anything about a Pokemon battle.
* How well Luna plays. Random picks won 1 to 3 of 12 games in smoke (b) while mix won 10 to 12, so
  Luna's choice really is the order and the A/B does see a bad decider; nothing here says a good one.
* The launcher's own questions in a real window (its lines are parsed by the tests, and its hidden
  prompt's quoting was run in cmd with a fixed text in place of the typing). If the hidden prompt
  should fail on some machine, `ladder.py` asks for the key itself.

## 9. Sol, and reading ahead (2026-10-13)

See `bench/notes/lines.md`: the arm `sol` (gpt-6-sol through the Responses API, the same question), the lines
arms (`mix-lines`, `luna-lines`, `sol-lines`), the window app's list and key field, and what was measured.
