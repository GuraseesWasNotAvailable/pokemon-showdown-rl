@echo off
title PSBot ladder Elo graph (cloud box)
rem Writes the rating graph on the training box, copies it to elo.box.html next to the old
rem laptop graph and opens it. "Net Elo.cmd test" does it for the local test run (elo.box-test.html,
rem not opened).
set "OUT=C:\Users\you\ps-ppo-play\runs\ladder\elo.box.html"
if /i "%~1"=="test" set "OUT=C:\Users\you\ps-ppo-play\runs\ladder\elo.box-test.html"
"C:\Program Files\Git\bin\bash.exe" C:/Users/you/ps-ppo-search/cloud/vm.sh ssh "bash ~/ladder/cloud/ladder_box.sh elo %1" > "%OUT%.part"
if errorlevel 1 (
  echo The graph could not be made: see the message above.
  pause
  exit /b 1
)
move /y "%OUT%.part" "%OUT%" > nul
echo Graph: %OUT%
if /i "%~1"=="test" (
  pause
  exit /b 0
)
start "" "%OUT%"
