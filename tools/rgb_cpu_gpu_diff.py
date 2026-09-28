# -*- coding: utf-8 -*-
"""rgb 两侧（CPU swscale vs GPU improc kernel）差异取证器（2026-09-27 深挖轮）。

问题：rgb 输出下 cpu / gpu / hybrid 三 ctx 的帧 md5 不一致；yuv420/gray
逐位一致。本工具回答三问：
1. 差多少：max|Δ| 分布、≥ 阈值计数、符号偏置（系统性还是对称）。
2. 差在哪：平坦色度区（U/V 近常数）还是结构区（边缘）。
3. hybrid 是谁：hybrid 每帧的 Δ 模式是否与 cpu↔gpu 差异二分一致
   （= 严格跟随其解码引擎，无第三种路径）。

用法:
    python tools/rgb_cpu_gpu_diff.py [video] [N]
默认 video=test.mp4（hevc），N=600。
"""
import os
import sys

os.environ.setdefault('DECORD_LIBRARY_PATH', 'D:/Repo/decord/build-dev')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np  # noqa: E402
from decord import VideoReader, cpu, gpu, hybrid  # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r'D:\Videos\racelog_test\test.mp4'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
B = 250

if not os.path.isfile(VIDEO):
    sys.exit('skip: ' + VIDEO + ' not available')


def read_frames(ctx):
    vr = VideoReader(VIDEO, ctx=ctx, output_format='rgb')
    vr.seek(0)
    out = []
    for s in range(0, N, B):
        out.append(vr.get_batch(list(range(s, min(s + B, N)))).asnumpy())
    del vr
    return np.concatenate(out) if len(out) > 1 else out[0]


fc = read_frames(cpu(0))
fg = read_frames(gpu(0))
fh = read_frames(hybrid(0))

dc = fg.astype(np.int16) - fc.astype(np.int16)   # gpu - cpu
dh = fh.astype(np.int16) - fc.astype(np.int16)   # hybrid - cpu
dg = fh.astype(np.int16) - fg.astype(np.int16)   # hybrid - gpu

absdc = np.abs(dc)
print(f'frames={fc.shape[0]} shape={fc.shape[1:]}')
for name, arr, ref in (('gpu-cpu', absdc, dc), ('hybrid-cpu', np.abs(dh), dh),
                       ('hybrid-gpu', np.abs(dg), dg)):
    m = arr >= 1
    print(f'{name}: max|Δ|={int(arr.max())}, ≥1: {int(m.sum())}, '
          f'≥4: {int((arr >= 4).sum())}, 符号和={int(ref[m].sum()) if m.any() else 0}')

# 平坦色度定位：取 yuv 帧的 U 平面，找 |Δ| 最大的像素并报告其局部 U 方差
vu = VideoReader(VIDEO, ctx=cpu(0), output_format='yuv420')
vu.seek(0)
yu = []
for s in range(0, N, B):
    yu.append(vu.get_batch(list(range(s, min(s + B, N)))).asnumpy())
del vu
yu = np.concatenate(yu) if len(yu) > 1 else yu[0]
h = fc.shape[1]
uplanes = yu[:, h:, :]   # (N, h/2, w) 交错 UV（Y+交错UV 布局）

flat_hits = 0
total_hits = 0
for i in range(fc.shape[0]):
    m = absdc[i] >= 2
    if not m.any():
        continue
    ys, xs, ch = np.nonzero(m)
    # 对应 U/V 采样位置（2x2 上采样 → 偶数行像素看 U，ch 交错看平面）
    uv = uplanes[i]
    uh, uw = uv.shape[:2]
    for y, x in zip(ys[::max(1, len(ys) // 64)], xs[::max(1, len(xs) // 64)]):
        uy = min(y // 2, uh - 1)
        ux = min(x // 2, uw - 1)
        y0, y1 = max(0, uy - 2), min(uh, uy + 3)
        x0, x1 = max(0, ux - 2), min(uw, ux + 3)
        total_hits += 1
        if int(uv[y0:y1, x0:x1].var()) <= 1:
            flat_hits += 1
print(f'|Δ|≥2 采样点落在平坦色度区（5x5 U 方差≤1）: {flat_hits}/{total_hits}')

# hybrid 二分性：每帧 hybrid-cpu 的 Δ 模式应 ≡ 0（cpu 帧或两侧同帧）或 ≡ gpu-cpu
bip = 0
for i in range(fc.shape[0]):
    hc = absdc[i].max()
    hh = np.abs(dh[i]).max()
    hg = np.abs(dg[i]).max()
    if hh == 0 or hg == 0 or (hc <= 3 and hh <= 3):
        bip += 1
print(f'hybrid 帧严格二分（每帧三选一全零距离）: {bip}/{fc.shape[0]}')
