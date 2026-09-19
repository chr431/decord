@echo off
rem ASCII-only: Chinese comments break cmd under GBK codepage.
rem Build the decord fork AND deploy the DLL into site-packages (with md5
rem check) so measurements never silently use a stale DLL.
rem
rem 2026-09-19 discipline round:
rem  - deploy used to be a manual `cp`; forgetting it meant measuring the OLD
rem    dll after editing C++ (silent wrong-result). Now build+deploy is one
rem    command and tools/env_doctor.py verifies md5 afterwards.
rem  - CMAKE_BUILD_TYPE used to silently drift (Debug vs Release). It is now
rem    explicit and overridable, and defaults to Release.
call "C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
if "%FFMPEG_DIR%"=="" set FFMPEG_DIR=D:/Software/ffmpeg-n9.0-latest-win64-gpl-shared-9.0
if "%CMAKE_BUILD_TYPE%"=="" set CMAKE_BUILD_TYPE=Release
cmake -B build-081fix -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DFFMPEG_DIR=%FFMPEG_DIR% .
if errorlevel 1 exit /b 1
cmake --build build-081fix --parallel %*
if errorlevel 1 exit /b 1

rem --- deploy into site-packages (kill stale python first) ---
taskkill /F /IM python.exe >nul 2>&1
set PYEXE=%RACELOG_PYTHON%
if "%PYEXE%"=="" set PYEXE=C:\Users\eric chen\AppData\Local\Programs\Python\python313\python.exe
"%PYEXE%" "D:\Repo\video_ocr_engine\tools\env_doctor.py" --deploy
if errorlevel 1 (
  echo.
  echo DEPLOY FAILED - see env_doctor output above.
  exit /b 1
)
