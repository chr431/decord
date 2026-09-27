#!/usr/bin/env bash
# hybrid 快速回归（2026-09-27 按主仓标准改造版）。
#
# 六套件：gpu / formats / md5 / stream / stride / lockstep。
# 默认串行逐套件跑（NVDEC 会话争用 C-01：并行跑 formats 会互相拖垮，
# 旧版三套件并行是隐藏 flake 源）；--parallel 释放三车道：
#   lane A = gpu        lane B = formats
#   lane C = md5 → stream → stride → lockstep（链式串行）
#
# 逐套件判定（PASS/FAIL + 退出码），聚合退出码非 0 即有失败。
# 用法: DECORD_LIBRARY_PATH=<dll目录> bash tests/run_fast.sh [N] [--parallel]
#   N 默认 600（lockstep 固定 600，与帧数无关）。
set -uo pipefail
cd "$(dirname "$0")/.."

N=600
PARALLEL=0
for a in "$@"; do
  case "$a" in
    --parallel) PARALLEL=1 ;;
    *) N="$a" ;;
  esac
done
export HYB_TEST_N="$N"
export DECORD_LIBRARY_PATH="${DECORD_LIBRARY_PATH:-D:/Repo/decord/build-081fix}"
TIMEOUT="${HYB_TEST_TIMEOUT:-900}"

# Windows Git Bash 无 /tmp 以外的稳定临时位；用固定子目录便于事后取证
LOGDIR="${TMPDIR:-/tmp}/run_fast.$$"
mkdir -p "$LOGDIR"

run_suite() {  # run_suite <name> <cmd...>
  local name="$1"; shift
  timeout "$TIMEOUT" "$@" > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "PASS $name"
  elif [ $rc -eq 124 ]; then
    echo "FAIL $name (timeout ${TIMEOUT}s) — $LOGDIR/$name.log"
  else
    echo "FAIL $name (exit $rc) — $LOGDIR/$name.log"
  fi
  return $rc
}

run_lane_c() {
  run_suite md5     python tests/test_hybrid.py --n "$N";       c1=$?
  run_suite stream  python tests/test_hybrid_stream.py;         c2=$?
  run_suite stride  python tests/test_hybrid_stride.py;         c3=$?
  run_suite lockstep python tests/test_hybrid_lockstep.py 600;  c4=$?
  return $(( c1 | c2 | c3 | c4 ))
}

FAILS=0
if [ "$PARALLEL" -eq 1 ]; then
  echo "== run_fast: N=$N, 3 lanes parallel (opt-in; NVDEC 争用自担) =="
  run_suite gpu     python tests/test_hybrid_gpu.py &  P1=$!
  run_suite formats python tests/test_hybrid_formats.py &  P2=$!
  run_lane_c &  P3=$!
  wait $P1; [ $? -ne 0 ] && FAILS=$((FAILS+1))
  wait $P2; [ $? -ne 0 ] && FAILS=$((FAILS+1))
  wait $P3; [ $? -ne 0 ] && FAILS=$((FAILS+1))
else
  echo "== run_fast: N=$N, serial =="
  run_suite gpu     python tests/test_hybrid_gpu.py      || FAILS=$((FAILS+1))
  run_suite formats python tests/test_hybrid_formats.py || FAILS=$((FAILS+1))
  run_lane_c || FAILS=$((FAILS+1))
fi

echo "== run_fast: $FAILS 个套件组失败；日志在 $LOGDIR =="
exit $(( FAILS > 0 ? 1 : 0 ))
