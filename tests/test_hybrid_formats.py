# -*- coding: utf-8 -*-
"""hybrid 输出格式/ROI 对齐验证。

用法: python tests/test_hybrid_formats.py
覆盖: {rgb, gray, yuv420} x {hevc, h264, av1} 全帧比对
      + yuv420/gray 的 ROI（字幕带状区域）比对 + seek 抽查

判据（2026-09-27 RGB 深挖轮定标，tools/rgb_cpu_gpu_diff.py 取证）：
- yuv420 / gray：与 cpu 逐字节一致（bit-exact）。
- rgb：CPU 路径 swscale 定点快路 vs GPU 路径 improc kernel 全精度
  float+roundf，固有算术差异 |Δ|max=3、≥4 计数为 0、系统性 +1~+2、
  只出现在平坦色度区（算术差异非结构性，hybrid 帧严格二分 =cpu 或
  =gpu 同侧帧）。据此设硬回归上界 RGB_MAX_TOL=3：越界即 FAIL——
  上界被突破说明是新的结构性差异（chroma siting / 矩阵回归），不是
  已知的舍入差。
"""
import os
import sys

os.environ.setdefault('DECORD_LIBRARY_PATH', 'D:/Repo/decord/build-081fix')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np  # noqa: E402
from decord import VideoReader, cpu, hybrid  # noqa: E402

_VIDEO_DIR = 'D:/Videos/racelog_test'
if __name__ == '__main__' and not os.path.isdir(_VIDEO_DIR):
    sys.exit('skip: test videos not available on this machine: ' + _VIDEO_DIR)
VIDS = [r'D:\Videos\racelog_test\test.mp4',
        r'D:\Videos\racelog_test\test3.mp4',
        r'D:\Videos\racelog_test\test6.mp4']
CODECS = ['hevc', 'h264', 'av1']
N = int(os.environ.get('HYB_TEST_N', '600'))
BLOCK = 250
ROI = (100, 200, 1800, 700)   # 字幕带状区域（半开 -> decord 闭区间转换）
RGB_MAX_TOL = 3               # 硬回归上界（见模块 docstring）


def bitexact_compare(path, fmt):
    """逐位一致比对。返回 (ok, detail)。"""
    vc = VideoReader(path, ctx=cpu(0), output_format=fmt, num_threads=12)
    vh = VideoReader(path, ctx=hybrid(0), output_format=fmt)
    total = min(N, len(vc))
    vc.seek(0); vh.seek(0)
    bad = -1
    matched = 0
    for s in range(0, total, BLOCK):
        e = min(s + BLOCK, total)
        bc = vc.get_batch(list(range(s, e))).asnumpy()
        bh = vh.get_batch(list(range(s, e))).asnumpy()
        if bc.shape != bh.shape:
            return False, f'SHAPE MISMATCH cpu{bc.shape} hyb{bh.shape}'
        for i in range(e - s):
            if bc[i].shape == bh[i].shape and bool((bc[i] == bh[i]).all()):
                matched += 1
            elif bad < 0:
                bad = s + i
        del bc, bh
    del vc, vh
    if bad < 0:
        return True, None
    return False, f'{matched}/{total} 一致, 首错帧 {bad}'


def rgb_tol_compare(path):
    """rgb 有界差异比对：max|Δ| ≤ RGB_MAX_TOL，统计差异面。返回 (ok, detail)。"""
    vc = VideoReader(path, ctx=cpu(0), output_format='rgb', num_threads=12)
    vh = VideoReader(path, ctx=hybrid(0), output_format='rgb')
    total = min(N, len(vc))
    vc.seek(0); vh.seek(0)
    gmax = 0            # 全局 max|Δ|
    ndiff = 0           # |Δ|≥1 像素计数
    bias_sum = 0.0      # 有差异像素的符号和（系统性 +1~+2 的证据面）
    worst = -1          # 最大差异所在帧
    for s in range(0, total, BLOCK):
        e = min(s + BLOCK, total)
        bc = vc.get_batch(list(range(s, e))).asnumpy().astype(np.int16)
        bh = vh.get_batch(list(range(s, e))).asnumpy().astype(np.int16)
        if bc.shape != bh.shape:
            return False, f'SHAPE MISMATCH cpu{bc.shape} hyb{bh.shape}'
        for i in range(e - s):
            # 逐帧累积（整批 bool 索引会物化 ~1.7GiB 临时数组，2026-09-28
            # 首跑实测 OOM）
            d = bh[i] - bc[i]
            a = np.abs(d)
            fm = int(a.max()) if a.size else 0
            if fm > gmax:
                gmax = fm
                worst = s + i
            m = a >= 1
            nd = int(m.sum())
            ndiff += nd
            if nd:
                bias_sum += float(d[m].sum())
            del d, a, m
        del bc, bh
    del vc, vh
    if gmax > RGB_MAX_TOL:
        return False, (f'max|Δ|={gmax} > {RGB_MAX_TOL}（结构性差异回归，'
                       f'worst=帧{worst}；已知固有差异上界 3，见 docstring）')
    mean_bias = (bias_sum / ndiff) if ndiff else 0.0
    return True, (f'max|Δ|={gmax} ≤ {RGB_MAX_TOL}, |Δ|≥1 像素 {ndiff}, '
                  f'平均符号偏置 {mean_bias:+.2f}, worst=帧{worst}')


def roi_compare(path, fmt):
    """ROI 比对 + seek(1000) 抽查（闭区间转换）。返回 (ok, detail)。"""
    roi = (ROI[0], ROI[1], ROI[2] - 1, ROI[3] - 1)
    vc = VideoReader(path, ctx=cpu(0), output_format=fmt, num_threads=12, roi=roi)
    vh = VideoReader(path, ctx=hybrid(0), output_format=fmt, roi=roi)
    total = min(600, len(vc))
    vc.seek(0); vh.seek(0)
    bad = -1
    matched = 0
    for s in range(0, total, BLOCK):
        e = min(s + BLOCK, total)
        bc = vc.get_batch(list(range(s, e))).asnumpy()
        bh = vh.get_batch(list(range(s, e))).asnumpy()
        if bc.shape != bh.shape:
            return False, f'SHAPE MISMATCH cpu{bc.shape} hyb{bh.shape}'
        for i in range(e - s):
            if bc[i].shape == bh[i].shape and bool((bc[i] == bh[i]).all()):
                matched += 1
            elif bad < 0:
                bad = s + i
        del bc, bh
    vh.seek_accurate(1000); vc.seek_accurate(1000)
    m = bool((vh[1000].asnumpy() == vc[1000].asnumpy()).all())
    del vc, vh
    if bad < 0 and m:
        return True, None
    return False, f'{matched}/{total} 一致, 首错 {bad}, seek {"对" if m else "错"}'


ok = True
for i, v in enumerate(VIDS):
    for fmt in ('yuv420', 'gray'):
        r, d = bitexact_compare(v, fmt)
        print(f'[{CODECS[i]}] {fmt:7s} 全帧: {"PASS" if r else "FAIL: " + str(d)}', flush=True)
        ok &= r
    r, d = rgb_tol_compare(v)
    print(f'[{CODECS[i]}] rgb    全帧: {"PASS" if r else "FAIL"} {d}', flush=True)
    ok &= r
    for fmt in ('yuv420', 'gray'):
        r, d = roi_compare(v, fmt)
        print(f'[{CODECS[i]}] {fmt:7s} ROI : {"PASS" if r else "FAIL: " + str(d)}', flush=True)
        ok &= r
print('ALL PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
