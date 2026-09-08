@echo off
rem Build the decord pip wheel.
rem vcvars provides cl/ninja; FFMPEG_DIR points at the BtbN ffmpeg-n9.0
rem shared build (same tree as release.yml CI). Extra args pass through
rem to pip wheel.
call "C:\Program Files\Microsoft Visual Studio8\Insiders\VC\Auxiliary\Buildcvars64.bat" >nul
cd /d %~dp0
if "%FFMPEG_DIR%"=="" set FFMPEG_DIR=D:/Repo/decord-release-dl/ffmpeg9/ffmpeg-n9.0-latest-win64-gpl-shared-9.0
python -m pip wheel . --no-deps --no-build-isolation -w dist %*