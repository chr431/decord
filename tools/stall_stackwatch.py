# -*- coding: utf-8 -*-
"""停滞栈取证（2026-09-28 R3-1 收官）：监听目标 stderr 的
[dump@eof-retry-probe] 标记行（FATAL 自旋期间每 ~2048 次重试一帧），
见标即 py-spy 采栈——拿到停滞期间 lander/CU worker/消费者的原生栈。

用法：python tools/stall_stackwatch.py [budget] [rounds]
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r'C:\Users\eric chen\AppData\Local\Programs\Python\python313\python.exe'
SPY = r'C:\Users\eric chen\AppData\Local\Programs\Python\python313\Scripts\py-spy.exe'
BUDGET = sys.argv[1] if len(sys.argv) > 1 else '10240'
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

env = dict(os.environ)
env['DECORD_LIBRARY_PATH'] = str(ROOT / 'build-081fix')
env['DECORD_EOF_RETRY_MAX'] = BUDGET

for rnd in range(1, ROUNDS + 1):
    rd = ROOT / 'tmp_stackwatch' / f'r{rnd}'
    rd.mkdir(parents=True, exist_ok=True)
    outf = open(rd / 'out.log', 'wb')
    t0 = time.time()
    proc = subprocess.Popen(
        [PY, 'tests/test_hybrid_stream.py'], cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    dumps = [0]

    def reader():
        for raw in proc.stdout:
            outf.write(raw)
            outf.flush()
            if b'dump@eof-retry-probe' in raw:
                i = dumps[0]
                dumps[0] += 1
                with open(rd / f'stall_{i:02d}.txt', 'wb') as df:
                    try:
                        subprocess.run(
                            [SPY, 'dump', '--native', '--pid', str(proc.pid)],
                            stdout=df, stderr=subprocess.DEVNULL, timeout=15)
                    except Exception:
                        pass
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    rc = proc.wait()
    th.join(timeout=5)
    outf.close()
    print(f'r{rnd}: rc={rc} dur={time.time()-t0:.0f}s dumps={dumps[0]}',
          flush=True)
    if dumps[0] > 0:
        print(f'  -> 栈样本在 {rd}', flush=True)
        break
