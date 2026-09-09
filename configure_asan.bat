@echo off
rem Configure + build an ASAN-instrumented local dev build (build-asan).
rem Diagnostic only (hybrid close race / memory bugs). NOT for release.
rem ASCII-only file: Chinese comments break cmd under GBK codepage.
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
cd /d %~dp0
cmake -B build-asan -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /W3 /GR /EHsc /fsanitize=address /Zi" "-DCMAKE_C_FLAGS=/DWIN32 /D_WINDOWS /W3 /fsanitize=address /Zi" -DFFMPEG_DIR=D:/Software/ffmpeg-n9.0-latest-win64-gpl-shared-9.0 .
if errorlevel 1 goto :eof
cmake --build build-asan --parallel
