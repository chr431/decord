# -*- coding: utf-8 -*-
"""hybrid_gpu vs gpu 吞吐对比（get_batch 不落宿主 —— 引擎/TRT 消费口径）。

用法: python tests/bench_hybrid_gpu.py [n]
"""
import os
import sys
import time

os.environ.setdefault('DECORD_LIBRARY_PATH',
                      'D:/Repo/decord/build-cuda13/Release')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from decord import VideoReader, gpu, hybrid_gpu, hybrid  # noqa: E402

_VIDEO_DIR = 'D:/Videos/racelog_test'
if __name__ == '__main__' and not os.path.isdir(_VIDEO_DIR):
    sys.exit('skip: test videos not available on this machine: ' + _VIDEO_DIR)
VIDS = [r'D:\Videos\racelog_test\test.mp4',
        r'D:\Videos\racelog_test\test3.mp4',
        r'D:\Videos\racelog_test\test6.mp4']
CODECS = ['hevc', 'h264', 'av1']
BLOCK = 250


def run(path, ctx, kw, n):
    vr = VideoReader(path, ctx=ctx, output_format='yuv420', **kw)
    vr.seek(0)
    t0, got = time.perf_counter(), 0
    while got < n:
        e = min(got + BLOCK, n)
        b = vr.get_batch(list(range(got, e)))
        got += b.shape[0]
        del b
    return got / (time.perf_counter() - t0)


n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
for i, v in enumerate(VIDS):
    r = {'gpu': run(v, gpu(0), {}, n),
         'hyb_gpu': run(v, hybrid_gpu(0), {}, n),
         'hyb_cpu': run(v, hybrid(0), {}, n)}
    fast = max(r['gpu'], r['hyb_gpu'])
    print(f'[{CODECS[i]}] gpu {r["gpu"]:.0f}  hybrid_gpu {r["hyb_gpu"]:.0f} '
          f'({r["hyb_gpu"]/r["gpu"]:.2f}x vs gpu)  hybrid(CPU-out) {r["hyb_cpu"]:.0f}',
          flush=True)
