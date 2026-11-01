@echo off
title PSBot lookahead on the Showdown ladder (this laptop), with predictions; A/B: games alternate between today's mix (the control) and round 2's set-up blend 8
rem Round 2 found no set-up rule that beats today's mix on the puzzles (setupfix bench/setup_puzzles/ROUND2.md).
rem This A/B keeps measuring its best candidate on the ladder, --setup-blend 8: +0.2 points (-0.8, +1.2) over
rem mix on the choose half, and the only rule that does not lose where the top player and the 5,000 ms engine
rem agree. Every game is "mix" or "mix-blend8", balanced; the end prints both arms' records and set-up sweep
rem losses with 95% intervals. Expect the intervals to cover 0 for a long time: a few tens of games per arm
rem cannot show a fraction of a point.
rem It runs the setupfix-play worktree (the blend arm is not in ps-ppo-predict). Run it INSTEAD of Look Play,
rem never at the same time. Look Stop still stops it (same stop file). Results: ps-ppo-setup-play\runs\ladder.
cd /d C:\Users\you\ps-ppo-setup-play
set PYTHONUTF8=1
set /p PS_USERNAME=Showdown username:
set GAMES=
set /p GAMES=How many games (Enter for 20):
if "%GAMES%"=="" set GAMES=20
"C:\Users\you\ps-ppo\.venv\Scripts\python.exe" ladder.py --games %GAMES% --arms mix,mix-blend8 --predictions --parallel 2 --helpers 3 --search-ms 500 --deadline 2.5 --stop-file "C:\Users\you\ps-ppo-predict\ladder.stop" --checkpoint "C:\Users\you\ps-ppo\runs\vm\checkpoints\learner_update_011850.pt"
echo.
echo Set-up sweeps per arm, all A/B games so far:
"C:\Users\you\ps-ppo\.venv\Scripts\python.exe" bench\setup_ladder_scan.py --dir runs\ladder --quiet --by-arm --arms mix,mix-blend8
echo.
pause
