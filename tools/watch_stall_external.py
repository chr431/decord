# -*- coding: utf-8 -*-
"""停滞外部观察台（2026-09-28 R3-1 续：零侵扰取证）。

进程内任何插桩（打印/线程/GIL）都会避开时序竞态（夜间轮已证），
本脚本完全在**目标进程外**观察：
  1. 每秒 py-spy dump --native：消费者/worker 线程的原生栈快照
  2. 每秒 psutil 线程 CPU 时间：停滞期线程忙/闲判别（忙=慢解码，
     闲=阻塞等待——两种病理的修法完全不同）
  3. 目标 stdout 逐行进度时间戳（停滞窗口定位）

用法：python tools/watch_stall_external.py [budget] [runs]
目标 = tests/test_hybrid_stream.py（失败原位套件）。
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
PY = r'C:\Users\eric chen\AppData\Local\Programs\Python\python313\python.exe'
SPY = r'C:\Users\eric chen\AppData\Local\Programs\Python\python313\Scripts\py-spy.exe'
BUDGET = sys.argv[1] if len(sys.argv) > 1 else '50000'
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = Path(os.environ.get('TEMP', '/tmp')) / 'stall_watch'
OUT.mkdir(exist_ok=True)

env = dict(os.environ)
env['DECORD_LIBRARY_PATH'] = str(ROOT / 'build-dev')
env['DECORD_EOF_RETRY_MAX'] = BUDGET

for run in range(1, RUNS + 1):
    rd = OUT / f'run{run}'
    rd.mkdir(exist_ok=True)
    logf = open(rd / 'stdout.log', 'wb')
    t0 = time.time()
    proc = subprocess.Popen(
        [PY, 'tests/test_hybrid_stream.py'], cwd=str(ROOT), env=env,
        stdout=logf, stderr=subprocess.STDOUT)
    p = psutil.Process(proc.pid)
    dumps = 0
    last_threads = None
    snap_f = open(rd / 'threads.tsv', 'w', encoding='utf-8')
    snap_f.write('t\tnthreads\ttid_user_sys_delta_ms\tbusy_tids\n')
    while proc.poll() is None:
        try:
            th = p.threads()
        except psutil.Error:
            break
        line = f'{time.time()-t0:.1f}\t{len(th)}\t'
        if last_threads is not None:
            prev = {t.id: (t.user_time + t.system_time) for t in last_threads}
            deltas = []
            for t in th:
                base = prev.get(t.id)
                if base is not None:
                    d = (t.user_time + t.system_time) - base
                    if d > 0.002:   # >2ms CPU in 1s = 忙
                        deltas.append((t.id, round(d * 1000)))
            line += ','.join(f'{i}:{u:.0f}' for i, u in deltas[:12])
        snap_f.write(line + '\n')
        snap_f.flush()
        last_threads = th
        # 原生栈转储（外部读取）
        with open(rd / f'dump_{dumps:04d}.txt', 'w', encoding='utf-8') as df:
            subprocess.run([SPY, 'dump', '--native', '--pid', str(proc.pid)],
                           stdout=df, stderr=subprocess.DEVNULL, timeout=10)
        dumps += 1
        time.sleep(1.0)
    snap_f.close()
    logf.close()
    rc = proc.returncode
    print(f'run{run}: exit={rc} dur={time.time()-t0:.1f}s dumps={dumps}',
          flush=True)
    if rc != 0:
        print(f'  -> FAIL run captured in {rd}', flush=True)
print('WATCH DONE')
