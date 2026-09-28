# -*- coding: utf-8 -*-
"""停滞触发器（2026-09-28 R3-1 续）：飞行记录器检测到产出冻结时才做
py-spy 采栈（两级取证——附挂暴露最小化，只在停滞窗口内）。

用法：DECORD_HYBRID_FLIGHT=1 已设的目标进程 pid 由本脚本 spawn：
    python tools/stall_trigger.py [budget]
流程：spawn tests/test_hybrid_stream.py → tail decord_hybrid_flight.log
→ fc/fg/em 三产出连续冻结 ≥3s 且 pend/qpk>0 → py-spy dump --native ×3
→ 保留触发前后飞行日志窗口。
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r'C:\Users\eric chen\AppData\Local\Programs\Python\python313\python.exe'
SPY = r'C:\Users\eric chen\AppData\Local\Programs\Python\python313\Scripts\py-spy.exe'
BUDGET = sys.argv[1] if len(sys.argv) > 1 else '50000'
OUT = Path(os.environ.get('TEMP', '/tmp')) / 'stall_trig'
OUT.mkdir(exist_ok=True)

env = dict(os.environ)
env['DECORD_LIBRARY_PATH'] = str(ROOT / 'build-dev')
env['DECORD_HYBRID_FLIGHT'] = '1'
env['DECORD_EOF_RETRY_MAX'] = BUDGET

FLIGHT = ROOT / 'decord_hybrid_flight.log'
if FLIGHT.exists():
    FLIGHT.unlink()

t0 = time.time()
logf = open(OUT / 'stdout.log', 'wb')
proc = subprocess.Popen([PY, 'tests/test_hybrid_stream.py'], cwd=str(ROOT),
                        env=env, stdout=logf, stderr=subprocess.STDOUT)
rd = OUT / f'pid{proc.pid}'
rd.mkdir(exist_ok=True)

LINE_RE = re.compile(
    r'\s*(\S+) eq=(\d+) gops=(\d+) head=(-?\d+),(-?\d+),(-?\d+),(-?\d+) '
    r'pend=(-?\d+) crdy=(\d+) rdy=(\d+) qpk=(\d+) gfl=(-?\d+) cq=(-?\d+) '
    r'cp=(-?\d+) cud=(-?\d+)/(-?\d+)/(-?\d+) fc=(\d+) fg=(\d+) em=(\d+) '
    r'fe=(\d+) eof=(\d+)')

history = []          # (t, fc, fg, em, full_line)
in_freeze = False
freeze_key = None
episodes = 0
trig_dumped = 0
pos = 0
while proc.poll() is None:
    time.sleep(0.25)
    if not FLIGHT.exists():
        continue
    with open(FLIGHT, 'r', encoding='utf-8', errors='replace') as f:
        f.seek(pos)
        chunk = f.read()
        pos = f.tell()
    for ln in chunk.splitlines():
        m = LINE_RE.match(ln)
        if not m:
            continue
        t = float(m.group(1))
        fc, fg, em = int(m.group(18)), int(m.group(19)), int(m.group(20))
        pend, qpk = int(m.group(8)), int(m.group(11))
        history.append((t, fc, fg, em, pend, qpk, ln))
    history = history[-64:]
    if len(history) >= 12 and not in_freeze:
        win = history[-12:]
        frozen = (win[0][1:4] == win[-1][1:4]
                  and all(w[1:4] == win[0][1:4] for w in win))
        work = win[-1][4] > 0 or win[-1][5] > 0
        if frozen and work:
            in_freeze = True
            freeze_key = win[-1][1:4]
            episodes += 1
            print(f'FREEZE#{episodes} t+{time.time()-t0:.1f}s '
                  f'flight_t={win[-1][0]:.1f} fc={win[-1][1]} '
                  f'fg={win[-1][2]} em={win[-1][3]} '
                  f'pend={win[-1][4]} qpk={win[-1][5]}', flush=True)
            (rd / 'flight_window.txt').write_text(
                '\n'.join(w[6] for w in history), encoding='utf-8')
    if in_freeze:
        if history and history[-1][1:4] != freeze_key:
            in_freeze = False   # 输出恢复：重武装（覆盖下一冻结 episode）
        elif trig_dumped < 30:
            with open(rd / f'stack_{trig_dumped:02d}.txt', 'w',
                      encoding='utf-8', errors='replace') as df:
                subprocess.run([SPY, 'dump', '--native', '--pid', str(proc.pid)],
                               stdout=df, stderr=subprocess.DEVNULL, timeout=15)
            trig_dumped += 1

# 收尾：全量飞行日志快照
try:
    import shutil
    shutil.copy(FLIGHT, rd / 'flight_full.log')
except OSError:
    pass
logf.close()
print(f'DONE rc={proc.returncode} dur={time.time()-t0:.1f}s '
      f'episodes={episodes} dumps={trig_dumped}', flush=True)
