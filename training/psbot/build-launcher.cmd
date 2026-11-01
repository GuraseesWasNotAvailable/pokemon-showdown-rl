@echo off
rem Rebuilds PSBot.exe from launcher\PSBot.cs and embeds the current psbot.js.
"%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /target:winexe /optimize+ /out:"%~dp0PSBot.exe" /r:System.Web.Extensions.dll /r:System.Windows.Forms.dll /resource:"%~dp0psbot.js",psbot.js "%~dp0launcher\PSBot.cs"
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo Built %~dp0PSBot.exe
