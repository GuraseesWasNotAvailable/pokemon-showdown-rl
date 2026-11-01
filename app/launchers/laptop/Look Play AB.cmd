@echo off
title PSBot ladder A/B on this laptop: net, mix and mix-deep, one per game (experiment ab3-net-mix-deep-024381)
rem The pre-registered A/B (scratchpad better\PLAN.md, step 1, with the owner's third arm): every game is
rem played by one of three arms, balanced over this experiment's own games (results.jsonl "experiment"):
rem   net       the network alone
rem   mix       today's network plus Foul Play's search, 500 ms per world, answer within 2.5 s
rem   mix-deep  mix with about 5 times the search: 1250 ms per world (2.5x) on twice the sampled opponent
rem             teams, answer within 7.5 s. A decision takes about 6 s: one 5 s tick of the battle timer,
rem             where every turn gives 10 s back, so the clock does not fall. If the timer ever shows under
rem             90 s, or a decision takes over 9 s, that game goes on with mix's budget (a console line).
rem 500 games per arm, one interim look at 250 per arm; bench\ab_report.py prints the verdict after every run.
rem Checkpoint 024381 (release r5: the observation is built as it was trained, hybrid.PolicyNet).
rem Every decision note also keeps the network's own score of the position (v_exp, phi, win_est).
rem
rem What it needs: 3 games at a time on 4 search helpers inside WSL. At most 3 searches run at once (one
rem per game, one core each), as in today's 3-game mix run; the deep arm holds its core longer, not more
rem cores. So about 4 of the 22 threads at the worst moment (3 helpers and the network on the GPU). Memory:
rem about 2 to 3 GB for the Windows side, and inside WSL 4 helpers of up to about 350 MB each with deep
rem searches (measured on the box: 257 to 343 MB at the most), so about 1.5 GB for WSL. On the box, mix
rem kept its iterations with deep searches beside it (+3.5% and -0.1%); on this laptop,
rem bench\ab_deep_measure.py checks it (offered below, about 4 minutes), and every game logs it too
rem (ab_report's "compute, paired" line). Start with 8 GB free.
rem
rem One bot per account: run it INSTEAD of the old Look Play and of the PSBot Lookahead app, never at the
rem same time. It refuses to start while another ladder.py plays on this computer, and ladder.py itself
rem takes the same account lock as the app. Look Stop still stops it (same stop file).
cd /d C:\Users\you\ps-ppo-predict
set PYTHONUTF8=1
set PY="C:\Users\you\ps-ppo\.venv\Scripts\python.exe"

set OTHERS=
for /f %%n in ('powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'ladder\.py' -and $_.CommandLine -notmatch '--graph|--history' }).Count"') do set OTHERS=%%n
if "%OTHERS%"=="" goto othersok
if "%OTHERS%"=="0" goto othersok
echo.
echo Another ladder.py is playing on this computer: the PSBot Lookahead app or a Look Play window.
echo Stop it first (Stop in the app, or Look Stop), then start this again. Two bots on one account
echo answer the same games, and the A/B would mix two programs' games.
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

set /p PS_USERNAME=Showdown username:
set GAMES=
set /p GAMES=How many games (Enter for 30):
if "%GAMES%"=="" set GAMES=30
set MEASURE=
set /p MEASURE=Measure mix's search with deep searches beside it first, about 4 minutes (y/N):
if /i "%MEASURE%"=="y" %PY% bench\ab_deep_measure.py --quick
%PY% ladder.py --games %GAMES% --arms net,mix,mix-deep --experiment ab3-net-mix-deep-024381 --predictions --parallel 3 --helpers 4 --search-ms 500 --deadline 2.5 --deep-search-ms 1250 --deep-worlds 2 --deep-deadline 7.5 --checkpoint "C:\Users\you\ps-ppo\runs\vm\checkpoints\learner_update_024381.pt"
echo.
echo The A/B so far (all of this experiment's games):
%PY% bench\ab_report.py --experiment ab3-net-mix-deep-024381 --mix-iterations-ref 538000
echo.
pause
