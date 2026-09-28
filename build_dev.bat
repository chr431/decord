@echo off
rem Build-only helper (no deploy): same as rebuild_dev.bat minus the
rem site-packages deploy + taskkill, so wheel-state tests keep running.
rem Configures first if build-dev has no cache (fresh tree / after rename).
call "C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\Repo\decord
if "%FFMPEG_DIR%"=="" set FFMPEG_DIR=D:/Software/ffmpeg-n9.0-latest-win64-gpl-shared-9.0
if not exist build-dev\CMakeCache.txt (
  cmake --preset dev
  if errorlevel 1 exit /b 1
)
cmake --build --preset dev --parallel
