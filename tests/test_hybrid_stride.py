# -*- coding: utf-8 -*-
"""原生 stride（等差采样路径 + discard-PTS）正确性与性能验证。

正确性判据：等差采样帧与全量解码取同索引帧逐字节一致（discard 只作用
于输出转换层，不影响解码/参考链）。
"""
import hashlib
import os
import sys
import time

os.environ.setdefault('DECORD_LIBRARY_PATH',
                      'D:/Repo/decord/build-cuda13/Release')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from decord import VideoReader, gpu, hybrid_gpu  # noqa: E402

_VIDEO_DIR = 'D:/Videos/racelog_test'
if __name__ == '__main__' and not os.path.isdir(_VIDEO_DIR):
    sys.exit('skip: test videos not available on this machine: ' + _VIDEO_DIR)
VIDS = [(r'D:\Videos\racelog_test\test.mp4', 'hevc'),
        (r'D:\Videos\racelog_test\test3.mp4', 'h264'),
        (r'D:\Videos\racelog_test\test6.mp4', 'av1')]
ROI = (841, 994, 949, 1026)
N = 2000
STRIDE = 8


def md5s(block):
    return [hashlib.md5(f.tobytes()).hexdigest() for f in block]


def full_frames(vr, n):
    out = []
    for s in range(0, n, 250):
        e = min(s + 250, n)
        out.extend(md5s(vr.get_batch(list(range(s, e))).asnumpy()))
    return out


def stride_frames(vr, n, stride):
    frames = list(range(0, n, stride))
    out = []
    B = 64
    for s in range(0, len(frames), B):
        e = min(s + B, len(frames))
        out.extend(md5s(vr.get_batch(frames[s:e]).asnumpy()))
    return out


ok = True
for v, codec in VIDS:
    for ctx_name, ctx, kw in (('gpu', gpu(0), {}),
                              ('hybrid', hybrid_gpu(0), {})):
        vr = VideoReader(v, ctx=ctx, output_format='yuv420', roi=ROI, **kw)
        vr.seek(0)
        full = full_frames(vr, N)
        vr.seek(0)
        st = stride_frames(vr, N, STRIDE)
        exp = full[::STRIDE]
        match = sum(1 for a, b in zip(st, exp) if a == b)
        ok &= match == len(exp)
        print(f'[{codec}] {ctx_name}: stride采样 {match}/{len(exp)} 与全量一致', flush=True)
        del vr

# 性能：test5 长视频 truth ROI
V5 = r'D:\Videos\racelog_test\test5.mp4'
ROI5 = (843, 993, 948, 1025)
FS, FE = 362, 7585
frames = list(range(FS, FE, STRIDE))
for ctx_name, ctx in (('gpu', gpu(0)), ('hybrid', hybrid_gpu(0))):
    vr = VideoReader(V5, ctx=ctx, output_format='yuv420', roi=ROI5)
    t0, got = time.perf_counter(), 0
    B = 64
    while got < len(frames):
        e = min(got + B, len(frames))
        vr.get_batch(frames[got:e])
        got += e - got
    dt = time.perf_counter() - t0
    print(f'[perf test5] {ctx_name}: {got} 采样帧 {dt:.1f}s '
          f'= {got/dt:.0f} fps (sampled) / {got*STRIDE/dt:.0f} fps (源帧等效)', flush=True)
    del vr
print('STRIDE TEST', 'ALL PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
