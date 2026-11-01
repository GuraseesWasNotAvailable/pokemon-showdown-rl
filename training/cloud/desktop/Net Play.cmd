@echo off
title PSBot on the Showdown ladder (cloud box)
rem Opens the bot's window on the training box: a tmux session called "ladder" in which
rem ~/ladder/cloud/ladder_box.sh runs ladder.py. When the bot is not running yet, it asks there for
rem the Showdown username, the number of games and the password (typed on the box, not shown, not
rem stored). When it already runs, this shows it playing. Closing this window (or Ctrl+B then D)
rem leaves the bot playing. "Net Play.cmd test" plays a local test server on the box instead.
"C:\Program Files\Git\bin\bash.exe" C:/Users/you/ps-ppo-search/cloud/vm.sh tty "bash ~/ladder/cloud/ladder_box.sh play %1"
echo.
echo Left the bot's window. If the bot was playing, it still is: Net Play shows it again, Net Stop stops it.
pause
