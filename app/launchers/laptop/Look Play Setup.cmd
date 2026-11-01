@echo off
title PSBot lookahead on the Showdown ladder (this laptop), with predictions, the veto fix and the set-up defer (unproven: measure it first with Look Play Setup AB)
cd /d C:\Users\you\ps-ppo-predict
set PYTHONUTF8=1
set /p PS_USERNAME=Showdown username: 
set GAMES=
set /p GAMES=How many games (Enter for 20): 
if "%GAMES%"=="" set GAMES=20
"C:\Users\you\ps-ppo\.venv\Scripts\python.exe" ladder.py --games %GAMES% --player mix --predictions --parallel 2 --helpers 3 --search-ms 500 --deadline 2.5 --veto-fix --setup-defer --checkpoint "C:\Users\you\ps-ppo\runs\vm\checkpoints\learner_update_011850.pt"
echo.
pause
