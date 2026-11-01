@echo off
title PSBot lookahead on the Showdown ladder (this laptop), with predictions; A/B: games alternate between today's mix (the control) and mix-counter
rem Round 3 (branch counterfix): 7 of this morning's 11 set-up sweep losses had a counter on the team that
rem was never used (phazing, Haze, Encore, Taunt, Yawn, sleep), and 12 of their 45 decisions hit the stat the
rem opponent had just boosted. The arm "mix-counter" is today's mix plus --setup-counter (in a set-up position,
rem our active Pokemon's own counter when it can work; since the review no switch to a benched one) and
rem --setup-other-side (an attack into a boosted defence gives way to one on the other defence that does at
rem least as much, accuracy counted). Every game is "mix" or "mix-counter", balanced from this A/B's own games
rem (results.jsonl "experiment"); the end prints both arms' records, how often each rule fired, and the set-up
rem sweep losses per arm with 95%% intervals. 100 games per arm show only a large loss; see
rem bench\notes\counter_fixes.md for how to read it.
rem It runs the counterfix worktree (the arm is not in ps-ppo-predict). Run it INSTEAD of Look Play, never at
rem the same time. Look Stop still stops it (same stop file). Results: ps-ppo-counter\runs\ladder.
cd /d C:\Users\you\ps-ppo-counter
set PYTHONUTF8=1
set /p PS_USERNAME=Showdown username:
set GAMES=
set /p GAMES=How many games (Enter for 20):
if "%GAMES%"=="" set GAMES=20
"C:\Users\you\ps-ppo\.venv\Scripts\python.exe" ladder.py --games %GAMES% --arms mix,mix-counter --predictions --parallel 2 --helpers 3 --search-ms 500 --deadline 2.5 --stop-file "C:\Users\you\ps-ppo-predict\ladder.stop" --checkpoint "C:\Users\you\ps-ppo\runs\vm\checkpoints\learner_update_011850.pt"
echo.
echo Set-up sweeps per arm, all A/B games so far:
"C:\Users\you\ps-ppo\.venv\Scripts\python.exe" bench\setup_ladder_scan.py --dir runs\ladder --quiet --by-arm --arms mix,mix-counter --experiment mix,mix-counter
echo.
pause
