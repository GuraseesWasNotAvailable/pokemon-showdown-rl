@echo off
title PSBot lookahead Elo graph
cd /d C:\Users\you\ps-ppo-predict
set PYTHONUTF8=1
"C:\Users\you\ps-ppo\.venv\Scripts\python.exe" ladder.py --graph
