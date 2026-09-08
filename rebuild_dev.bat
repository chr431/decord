@echo off
rem Incremental rebuild of the local dev build (build-081fix: Ninja + ffmpeg9).
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
cmake --build build-081fix --parallel %*