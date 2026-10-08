# -*- coding: utf-8 -*-
"""kick 竞态布局轮盘（夜间轮 2026-09-28；2026-10-08 参数化）：窗口位级
探针版。

时序竞态对二进制布局敏感（layout_roulette.py 同款方法论）：迭代注入
冷注释移位 .text 布局，纯 ninja 重建（不经 rebuild_dev.bat 的部署步），
每布局跑 N 次指定格 hybrid vs nvdec 位级对照，命中即停（保留布局 +
md5 + TR2 全量 trace 供取证）。

用法：python tools/_kick_roulette.py [起始移位] [每布局探针数] [探针轮]
                                  [video=...|start=...|win=...|roi=...]
  默认格 = hevc 晚起点 seek_acc(5000)+win(3000)（僵尸 kick 历史形态）；
  KEY=VAL 覆盖：video=<mp4 路径> start=<int> win=<int> roi=<(x1,y1,x2,y2)>
  例：... _kick_roulette.py 16 3 1 video=test6.mp4 start=0 win=3000
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r'C:\Users\eric chen\AppData\Local\Programs\Python\python313\python.exe'
PUMP = ROOT / 'src/video/hybrid/pump.cc'
MARK = '// LAYOUT-SHIFT:'

start = int(sys.argv[1]) if len(sys.argv) > 1 else 16
nprobe = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
CELL = {'video': r'D:\Videos\racelog_test\test6_hevc.mp4',
        'roi': '(841, 994, 949, 1026)', 'start': 5000, 'win': 3000}
for a in sys.argv[4:]:
    k, _, v = a.partition('=')
    if k in CELL:
        CELL[k] = int(v) if v.lstrip('-').isdigit() else v

PROBE = r'''
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from decord import VideoReader, gpu, hybrid
path = r"{video}"
roi = {roi}
START, N = {start}, {win}
def frames(ctx, use_win):
    vr = VideoReader(path, ctx=ctx, output_format="gray", roi=roi, num_threads=32)
    if use_win: vr.set_decode_window(N)
    vr.seek_accurate(START)
    out = []; got = 0
    while got < N:
        e = min(got+64, N)
        b = vr.get_batch(list(range(START+got, START+e)))
        out.append(b.asnumpy().copy()); got += b.shape[0]; del b
    vr.close()
    return np.concatenate(out)
base = frames(gpu(0), False)
a = frames(hybrid(0), True)
bad = [i for i in range(N) if (a[i] != base[i]).any()]
print("BAD", len(bad), (bad[0], bad[-1]) if bad else "")
'''.format(**CELL)


def dll_md5():
    return hashlib.md5(
        (ROOT / 'build-dev/decord.dll').read_bytes()).hexdigest()[:12]


env = dict(os.environ)
env['DECORD_LIBRARY_PATH'] = str(ROOT / 'build-dev')

for shift in range(start, start + 608, 16):
    src = PUMP.read_text(encoding='utf-8')
    lines = [l for l in src.split('\n') if not l.startswith(MARK)]
    idx = next(i for i, l in enumerate(lines)
               if l.startswith('bool HybridThreadedDecoder::Pop('))
    lines.insert(idx, MARK + ' ' + 'x' * shift)
    PUMP.write_text('\n'.join(lines), encoding='utf-8', newline='')
    r = subprocess.run(['cmd', '/c', 'build-dev\\_build_busy.bat'],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=600)
    if 'Linking' not in r.stdout:
        print(f'shift={shift}: BUILD FAIL\n{r.stdout[-600:]}', flush=True)
        sys.exit(1)
    md5 = dll_md5()
    fails = 0
    for _ in range(rounds):
        for _ in range(nprobe):
            rr = subprocess.run([PY, '-c', PROBE], env=env,
                                capture_output=True, text=True,
                                encoding='utf-8', errors='replace',
                                timeout=300)
            if 'BAD 0 ' not in rr.stdout:
                fails += 1
    print(f'shift={shift:3d} md5={md5} fails={fails}/{nprobe*rounds}', flush=True)
    if fails:
        out = ROOT / 'tmp_kick_probe'
        out.mkdir(exist_ok=True)
        (out / f'layout{shift}.log').write_text(
            f'shift={shift} md5={md5} fails={fails}/{nprobe*rounds}',
            encoding='utf-8')
        print(f'HIT: shift={shift} md5={md5} — pump.cc 布局保留勿动', flush=True)
        sys.exit(0)
print('no hit in range', flush=True)
