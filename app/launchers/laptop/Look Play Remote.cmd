@echo off
title PSBot ladder A/B: this laptop plays, the cloud box searches (experiment ab3-net-mix-deep-024381)
rem The pre-registered three-arm A/B (net, mix, mix-deep; bench\ab_report.py) with the search on the cloud box.
rem
rem This laptop keeps the Showdown connection (Showdown sees this home connection, never the box) and the network
rem on its GPU. Foul Play's search runs on the box, each helper started over ssh by ladder.py (--search-remote):
rem 6 games at a time on 7 helpers, mix 500 ms per world, mix-deep 1425 ms per world on twice the worlds, 4 at a
rem time, within 5 s (the box's calibrated settings, as much search as the laptop's own games had).
rem
rem cloud\remote_search.sh does the rest: no other bot on this account, the box started (when AWS has no capacity
rem for 15 minutes: play on the laptop with WSL as before, or quit), this code copied to the box, the round trip
rem measured, ladder.py (it asks for the Showdown password itself), the report, and at the end:
rem "Stop the cloud box now? [Y/n]" (stopped unless n is typed within 60 s; it costs about $7.58 an hour).
rem
rem One bot per account: run it INSTEAD of Look Play, Look Play AB and the PSBot Lookahead app. Look Stop stops it
rem (no new games, the ones in progress are finished).
cd /d C:\Users\you\ps-ppo-predict
set PYTHONUTF8=1
set "GITBASH=C:\Program Files\Git\bin\bash.exe"
if exist "%GITBASH%" goto run
echo Git Bash is not at %GITBASH%: it is needed for the box (cloud\vm.sh, ssh).
pause
exit /b 1
:run
"%GITBASH%" cloud/remote_search.sh
