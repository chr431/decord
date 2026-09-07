#!/usr/bin/env bash
# hybrid 快速回归:三套件并行 + 帧数降为 600(HYB_TEST_N 可覆盖)
# 用法: DECORD_LIBRARY_PATH=<dll目录> bash tests/run_fast.sh [N]
set -u
cd "$(dirname "$0")/.."
export HYB_TEST_N="${1:-600}"
export DECORD_LIBRARY_PATH="${DECORD_LIBRARY_PATH:-D:/Repo/decord/build-ff9/Release}"
sys_dir=$(cygpath -w "$PWD/python" 2>/dev/null || echo "$PWD/python")
run() { python "$1" ${2:-} 2>&1 | tail -3; }
echo "== run_fast: N=$HYB_TEST_N, 3 suites in parallel =="
run tests/test_hybrid_gpu.py > /tmp/t_gpu.log 2>&1 &
P1=$!
run tests/test_hybrid_formats.py > /tmp/t_fmt.log 2>&1 &
P2=$!
( HYB_TEST_N=$HYB_TEST_N python tests/test_hybrid.py --n "$HYB_TEST_N" > /tmp/t_md5.log 2>&1; python tests/test_hybrid_stream.py >> /tmp/t_md5.log 2>&1; python tests/test_hybrid_stride.py >> /tmp/t_md5.log 2>&1 ) &
P3=$!
wait $P1; R1=$?
wait $P2; R2=$?
wait $P3; R3=$?
echo "-- test_hybrid_gpu:"; tail -2 /tmp/t_gpu.log
echo "-- test_hybrid_formats:"; tail -2 /tmp/t_fmt.log
echo "-- md5/stream/stride:"; grep -E "md5一致|seek|ALL PASS|FAIL" /tmp/t_md5.log | tail -6
FAILS=$(( $(grep -c FAIL /tmp/t_gpu.log) + $(grep -c FAIL /tmp/t_fmt.log) + $(grep -c FAIL /tmp/t_md5.log) + $(grep -c "seek \[" /tmp/t_md5.log) ))
echo "== exit codes: $R1 $R2 $R3; FAIL mentions: $FAILS =="
