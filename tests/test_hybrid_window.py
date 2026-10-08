# -*- coding: utf-8 -*-
"""硬窗（set_decode_window）位级验证（窗口架构重做 2026-10-08 第七套件）。

背景：fork 自身此前对窗口正确性零覆盖（正确性完全由引擎仓的矩阵探针
承载）；重做后 fork 侧常驻本套件。每编码 × 早/晚起点 × {硬窗, 无窗,
纯 gpu} 三读法逐字节比对 + got==n + win_subs==0（结构性恒 0：窗模式
禁替补，静默换帧在架构上不可达）。
"""
import os
import sys

os.environ.setdefault('DECORD_LIBRARY_PATH',
                      'D:/Repo/decord/build-dev')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from decord import VideoReader, gpu, hybrid  # noqa: E402

_VIDEO_DIR = 'D:/Videos/racelog_test'
if __name__ == '__main__' and not os.path.isdir(_VIDEO_DIR):
    sys.exit('skip: test videos not available on this machine: ' + _VIDEO_DIR)
VIDS = [r'D:\Videos\racelog_test\test.mp4',
        r'D:\Videos\racelog_test\test3.mp4',
        r'D:\Videos\racelog_test\test6.mp4']
CODECS = ['hevc', 'h264', 'av1']
ROI = (100, 200, 1799, 699)   # decord 闭区间
# 早起点（窗=请求长）+ 晚起点（C-57 历史缺陷形态：seek 目标 ≥ 窗长）
GRID = [(0, 3000), (5000, 1000)]


def read(path, start, n, ctx, use_win):
    vr = VideoReader(path, ctx=ctx, output_format='gray', roi=ROI,
                     num_threads=16)
    if use_win:
        vr.set_decode_window(n)
    vr.seek_accurate(start)
    out = []
    got = 0
    while got < n:
        e = min(got + 64, n)
        b = vr.get_batch(list(range(start + got, start + e))).asnumpy()
        out.append(b.copy())
        got += b.shape[0]
        del b
    st = vr.hybrid_stats() or {}
    vr.close()
    import numpy as np
    return np.concatenate(out) if out else None, got, st.get('win_subs')


def main():
    fails = 0
    for path, codec in zip(VIDS, CODECS):
        if not os.path.isfile(path):
            print('skip missing', path)
            continue
        total = len(VideoReader(path))
        for start, win in GRID:
            n = min(win, total - start)
            if n <= 0:
                continue
            a, ga, sa = read(path, start, n, hybrid(0), True)
            b, gb, _ = read(path, start, n, hybrid(0), False)
            c, gc, _ = read(path, start, n, gpu(0), False)
            ok = (a is not None and a.shape == b.shape == c.shape
                  and (a == b).all() and (b == c).all()
                  and ga == gb == gc == n and sa == 0)
            if not ok:
                fails += 1
            print('[%s] start=%-5d win=%-4d %s win_subs=%s'
                  % (codec, start, n, 'PASS' if ok else 'FAIL', sa),
                  flush=True)
    if fails:
        sys.exit('WINDOW TEST FAILED: %d cells' % fails)
    print('WINDOW TEST ALL PASS')


if __name__ == '__main__':
    main()
