@echo off
rem ASCII-only: Chinese comments break cmd under GBK codepage.
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
cmake -B build-081fix -DFFMPEG_DIR=D:/Software/ffmpeg-n9.0-latest-win64-gpl-shared-9.0 .
cmake --build build-081fix --parallel %*
