@echo off
rem Builds "PSBot Lookahead.exe" (next to this file) from app\PSBotLookahead.cs with the C# compiler
rem that ships with Windows (.NET Framework 4). The exe runs ladder.py from this folder.
"%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /target:winexe /optimize+ /out:"%~dp0PSBot Lookahead.exe" /r:System.Web.Extensions.dll /r:System.Management.dll /r:System.Windows.Forms.dll /r:System.Drawing.dll "%~dp0app\PSBotLookahead.cs"
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo Built "%~dp0PSBot Lookahead.exe"
