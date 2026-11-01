@echo off
title Stop the PSBot ladder bot (cloud box)
rem Writes the stop file on the training box: the bot starts no new games, finishes the ones in
rem progress and stops. "Net Stop.cmd test" stops the local test run instead.
"C:\Program Files\Git\bin\bash.exe" C:/Users/you/ps-ppo-search/cloud/vm.sh ssh "bash ~/ladder/cloud/ladder_box.sh stop %1"
pause
