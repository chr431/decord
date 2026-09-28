# -*- coding: utf-8 -*-
"""布局轮盘驱动器 v2（2026-09-28 停滞取证）：时序竞态对二进制布局敏感
（同夜观察：注释级差异可使失败率 0% ↔ 60-75%）。迭代注入冷注释移位
.text 布局，重建（_build_busy.bat 纯 ninja——rebuild_dev.bat 的部署步
taskkill python 会杀死本驱动器）后以默认预算探针，命中即停。

用法：python tools/layout_roulette.py [起始移位] [每布局探针数]
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r'C:\Users\eric chen\AppData\Local\Programs\Python\python313\python.exe'
PUMP = ROOT / 'src/video/hybrid/pump.cc'
MARK = '// LAYOUT-SHIFT:'

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
nprobe = int(sys.argv[2]) if len(sys.argv) > 2 else 4
env = dict(os.environ)
env['DECORD_LIBRARY_PATH'] = str(ROOT / 'build-081fix')
# grace 修复后默认预算下 FATAL 不可能（在途宽限 10×）——用 1024 把
# 在途停滞的 FATAL 阈值拉回 ~10.4s，FATAL 即停滞探测器
env['DECORD_EOF_RETRY_MAX'] = '1024'

import hashlib


def dll_md5():
    return hashlib.md5(
        (ROOT / 'build-081fix/decord.dll').read_bytes()).hexdigest()[:12]


for shift in range(start, start + 608, 16):
    src = PUMP.read_text(encoding='utf-8')
    lines = [l for l in src.split('\n') if not l.startswith(MARK)]
    idx = next(i for i, l in enumerate(lines)
               if l.startswith('bool HybridThreadedDecoder::Pop('))
    lines.insert(idx, MARK + ' ' + 'x' * shift)
    PUMP.write_text('\n'.join(lines), encoding='utf-8', newline='')
    r = subprocess.run(['cmd', '/c', 'build-081fix\\_build_busy.bat'],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=600)
    if 'Linking' not in r.stdout:
        print(f'shift={shift}: BUILD FAIL\n{r.stdout[-600:]}', flush=True)
        sys.exit(1)
    md5 = dll_md5()
    fails = 0
    for i in range(nprobe):
        rr = subprocess.run([PY, 'tests/test_hybrid_stream.py'],
                            cwd=str(ROOT), env=env, capture_output=True,
                            text=True, encoding='utf-8', errors='replace',
                            timeout=300)
        if rr.returncode != 0:
            fails += 1
    print(f'shift={shift:3d} md5={md5} fails={fails}/{nprobe}', flush=True)
    if fails:
        out = ROOT / 'tmp_rate_probe'
        out.mkdir(exist_ok=True)
        (out / f'layout{shift}.log').write_text(
            'shift=%d md5=%s fails=%d/%d' % (shift, md5, fails, nprobe),
            encoding='utf-8')
        print(f'HIT: shift={shift} md5={md5} — pump.cc 布局保留勿动', flush=True)
        # 取证复跑：flight 记录器 + 停滞触发 py-spy（同布局，同检测预算）
        tr = subprocess.run([PY, '-u', 'tools/stall_trigger.py', '1024'],
                            cwd=str(ROOT), capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=600)
        print('trigger run:', tr.stdout[-800:], tr.stderr[-300:], flush=True)
        sys.exit(0)
print('no hit in range', flush=True)
