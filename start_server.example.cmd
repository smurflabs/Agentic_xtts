@echo off
rem XTTS-v2 API launcher. Copy to start_server.cmd (git-ignored) and fill in your settings.
rem Generate a key in PowerShell:
rem   $b = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b); -join ($b | % { '{0:x2}' -f $_ })
set XTTS_HOST=127.0.0.1
set XTTS_PORT=8020
set XTTS_WORKERS=2
set XTTS_API_KEY=change-me
set XTTS_VOICE_TTL_DAYS=30
set XTTS_FAST=1

rem Only needed when running as SYSTEM (e.g. a boot-time scheduled task): SYSTEM has no user
rem profile, so point it at the model caches of the account that downloaded the model.
rem set TTS_HOME=C:\Users\<you>\AppData\Local
rem set HF_HOME=C:\Users\<you>\.cache\huggingface
rem set HF_HUB_OFFLINE=1

cd /d %~dp0
:loop
venv\Scripts\python.exe server.py >> server.log 2>&1
echo [%date% %time%] server exited (code %errorlevel%), restarting in 10s >> server.log
timeout /t 10 /nobreak > nul
goto loop
