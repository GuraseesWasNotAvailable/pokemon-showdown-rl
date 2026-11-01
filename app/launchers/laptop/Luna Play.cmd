@echo off
title PSBot with a model of OpenAI deciding (Luna or Sol) on the Showdown ladder
rem Luna decides: every turn our bot works out what it does today (the network, Foul Play's search,
rem the read of the opponent), writes it down in plain words, and asks OpenAI's Decisions API
rem (model gpt-6-luna) ONE question: which of our legal actions to play. Luna's choice is played.
rem luna_player.py has the whole of it; ladder.py --help the flags.
rem
rem The first question picks the model: Luna (gpt-6-luna, about 0.15 s and a hundredth of a cent a
rem move), or Sol (gpt-6-sol through the Responses API: larger and slower, 1 to 5 s and about half
rem a cent a move, so about 15 to 20 cents a game). The second one turns on reading ahead: after
rem the search, its 3 favourite moves are each played 5 turns forward and the model sees how each
rem line ends (about 2 s more a move). What is said about Luna below holds for Sol too, with Sol's
rem 5 s in place of Luna's 1.2 s.
rem
rem THIS PLAYS RATED GAMES on the Showdown account typed in below. Nobody can promise a rating:
rem choice 2 (the default) alternates Luna with today's bot game by game, and the report after the
rem run says which of the two won more. 30 games show that it runs; a verdict needs about 250 games
rem per arm (bench\ab_report.py says so itself until then).
rem
rem Luna never costs a game: when no legal answer comes within 1.2 s (and never past the 2.5 s the
rem decision has), on an API error, a refusal, or a choice that is not legal, today's bot's move is
rem played; after 5 API failures in a row it is not asked for 60 s; and a blunder guard keeps
rem today's move when Luna's has under 1%% of the network and of the search while today's has over
rem 60%% of either (--luna-guard off turns that off). Each of these is counted and printed.
rem
rem The OpenAI key: taken from the environment variable OPENAI_API_KEY when it is set, else asked
rem here once, hidden. It stays in this window's memory: it is never written to a file, a log or a
rem command line, and it is cleared when the run ends. About 1,500 input tokens per decision at
rem $0.10 per million: a game of 40 decisions costs under one cent.
rem
rem What it needs: as Look Play AB, 3 games at a time on 4 search helpers inside WSL, and the
rem internet for the API. To search on the cloud box instead, add --search-remote vm to the two
rem ladder.py lines below (as Look Play Remote does).
rem
rem One bot per account: run it INSTEAD of Look Play, Look Play AB and the PSBot Lookahead app.
rem Luna Stop stops it (no new games, the games in progress are finished).
cd /d C:\Users\you\ps-ppo-luna
set PYTHONUTF8=1
set PY="C:\Users\you\ps-ppo\.venv\Scripts\python.exe"
set CKPT="C:\Users\you\ps-ppo\runs\vm\checkpoints\learner_update_024381.pt"

echo.
echo  Luna Play: a model of OpenAI (Luna or Sol) picks every move, with today's bot as its adviser.
echo  WARNING: this plays RATED games on the Showdown account you type in below.
echo  Its rating can go down as well as up. Close this window now if you do not want that.
echo.

set OTHERS=
for /f %%n in ('powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'ladder\.py' -and $_.CommandLine -notmatch '--graph|--history|--luna-check' }).Count"') do set OTHERS=%%n
if "%OTHERS%"=="" goto othersok
if "%OTHERS%"=="0" goto othersok
echo Another ladder.py is playing on this computer: the PSBot Lookahead app or a Look Play window.
echo Stop it first (Stop in the app, or Look Stop), then start this again. Two bots on one account
echo answer the same games.
echo.
pause
exit /b 1
:othersok

:memcheck
set FREEGB=
for /f %%m in ('powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)"') do set FREEGB=%%m
if "%FREEGB%"=="" goto memok
if %FREEGB% GEQ 8 goto memok
echo.
echo Only %FREEGB% GB of memory is free. The run wants 8 GB free: please close what you can
echo (browser tabs, other Claude sessions, games), then press Enter to check again.
set ANSWER=
set /p ANSWER=Or type GO and Enter to start anyway:
if /i "%ANSWER%"=="GO" goto memok
goto memcheck
:memok
if not "%FREEGB%"=="" echo %FREEGB% GB of memory free.

set ASKEDKEY=
if not "%OPENAI_API_KEY%"=="" goto keyok
echo.
echo The OpenAI API key is needed (platform.openai.com, API keys). What you type is not shown.
set ASKEDKEY=1
for /f "usebackq delims=" %%k in (`powershell -NoProfile -Command "$s = Read-Host -AsSecureString 'OpenAI API key'; $c = New-Object System.Net.NetworkCredential -ArgumentList 'k', $s; $c.Password"`) do set "OPENAI_API_KEY=%%k"
:keyok

echo.
set MODEL=
echo Which model picks the moves?
echo 1 = Luna (gpt-6-luna): about 0.15 s and a hundredth of a cent a move
echo 2 = Sol (gpt-6-sol): larger and slower, 1 to 5 s and about half a cent a move
set /p MODEL=Which (Enter for 1):
set ARM=luna
set NAME=Luna
if "%MODEL%"=="2" set ARM=sol
if "%MODEL%"=="2" set NAME=Sol
set CHECKMODEL=%ARM%
set AHEAD=
set /p AHEAD=Read ahead 5 turns before each move, about 2 s more a move (y/N):
if /i "%AHEAD%"=="y" set ARM=%ARM%-lines

echo.
%PY% ladder.py --luna-check --model %CHECKMODEL%
if errorlevel 1 goto checkfailed

echo.
set /p PS_USERNAME=Showdown username:
if "%PS_USERNAME%"=="" goto nouser
echo RATED games will be played on the Showdown account %PS_USERNAME%. Close this window now if that is not wanted.
set MODE=
echo 1 = %NAME% every game
echo 2 = A/B: %NAME% and today's bot alternate (shows whether %NAME% is better)
set /p MODE=Which (Enter for 2):
if "%MODE%"=="" set MODE=2
set GAMES=
set /p GAMES=How many games (Enter for 30):
if "%GAMES%"=="" set GAMES=30
if "%MODE%"=="1" goto lunaonly

%PY% ladder.py --games %GAMES% --arms mix,%ARM% --experiment %ARM%-vs-mix-024381 --predictions --parallel 3 --helpers 4 --search-ms 500 --deadline 2.5 --checkpoint %CKPT%
echo.
echo The A/B so far (all of this experiment's games; control mix, arm %ARM%):
%PY% bench\ab_report.py --experiment %ARM%-vs-mix-024381 --control mix --arms mix,%ARM%
goto summary

:lunaonly
%PY% ladder.py --games %GAMES% --player %ARM% --predictions --parallel 3 --helpers 4 --search-ms 500 --deadline 2.5 --checkpoint %CKPT%

:summary
echo.
echo %NAME% so far (every game of the arm %ARM% played from this folder):
%PY% bench\luna_report.py --arm %ARM%
goto done

:nouser
echo.
echo No Showdown username was typed: nothing was played.
goto done

:checkfailed
echo.
echo The check of %NAME% failed (the line above says why): nothing was played.
echo A key that was typed wrong: start this again and paste it once more.

:done
if "%ASKEDKEY%"=="1" set OPENAI_API_KEY=
echo.
pause
