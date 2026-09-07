# -*- coding: utf-8 -*-
"""hybrid ctx 快速验证 harness（低内存、多线程、流式比对）.

用法:
  python tests/test_hybrid.py                 # 3 codec x 2000 帧 + seek
  python tests/test_hybrid.py --video PATH --n 4000
  python tests/test_hybrid.py --stress 3      # 多 reader 交替压力(显存泄漏检查)
"""
import argparse
import hashlib
import os
import random
import sys
import time

os.environ.setdefault('DECORD_LIBRARY_PATH',
                      os.environ.get('DECORD_LIBRARY_PATH',
                                     r'D:/Repo/decord/build-cuda13/Release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from decord import VideoReader, cpu, gpu, hybrid  # noqa: E402

BLOCK = 250          # 分块大小: 内存 ~BLOCK*3MB (250->~0.8GB 对 1080p yuv420)
CPU_THREADS = 12     # 参考软解线程数(占满核, 之前 num_threads=0 只有 2 线程)


def frames_equal(a, b):
    return a.shape == b.shape and bool((a == b).all())


def verify(path, n, verbose=True):
    """hybrid vs cpu 分块流式全帧比对 + 随机 seek 抽查."""
    t0 = time.perf_counter()
    vc = VideoReader(path, ctx=cpu(0), output_format='yuv420',
                     num_threads=CPU_THREADS)
    vh = VideoReader(path, ctx=hybrid(0), output_format='yuv420')
    total = min(n, len(vc))
    matched = 0
    first_bad = -1
    vc.seek(0)
    vh.seek(0)
    for s in range(0, total, BLOCK):
        e = min(s + BLOCK, total)
        bc = vc.get_batch(list(range(s, e))).asnumpy()
        bh = vh.get_batch(list(range(s, e))).asnumpy()
        for i in range(e - s):
            if frames_equal(bc[i], bh[i]):
                matched += 1
            elif first_bad < 0:
                first_bad = s + i
        del bc, bh
    random.seed(7)
    seek_ok = []
    for t in random.sample(range(total, len(vh)), 3):
        vh.seek_accurate(t)
        vc.seek_accurate(t)
        seek_ok.append(frames_equal(vh[t].asnumpy(), vc[t].asnumpy()))
    del vc, vh
    dt = time.perf_counter() - t0
    if verbose:
        print(f'  {os.path.basename(path)}: {matched}/{total} md5一致 '
              f'(首错 {first_bad}), seek {"全对" if all(seek_ok) else seek_ok}, '
              f'{dt:.1f}s')
    return matched == total and all(seek_ok)


def bench(path, n):
    """cpu/gpu/hybrid 吞吐对比(分块取, 不做比对)."""
    def run(ctx, kw):
        vr = VideoReader(path, ctx=ctx, output_format='yuv420', **kw)
        vr.seek(0)
        t0 = time.perf_counter()
        got = 0
        while got < n:
            e = min(got + BLOCK, n)
            b = vr.get_batch(list(range(got, e)))
            got += b.shape[0]
        return got / (time.perf_counter() - t0)
    r = {'cpu': run(cpu(0), dict(num_threads=CPU_THREADS)),
         'gpu': run(gpu(0), {}),
         'hyb': run(hybrid(0), {})}
    fast = max(r['cpu'], r['gpu'])
    print(f'  {os.path.basename(path)}: cpu {r["cpu"]:.0f} gpu {r["gpu"]:.0f} '
          f'hyb {r["hyb"]:.0f} fps ({r["hyb"]/fast:.2f}x vs 最快)')


def stress(rounds):
    """多 reader 交替开关压力: 验证显存/内存无泄漏式增长(配合 nvidia-smi)."""
_VIDEO_DIR = 'D:/Videos/racelog_test'
if __name__ == '__main__' and not os.path.isdir(_VIDEO_DIR):
    sys.exit('skip: test videos not available on this machine: ' + _VIDEO_DIR)
    vids = [r'D:\Videos\racelog_test\test.mp4',
            r'D:\Videos\racelog_test\test3.mp4',
            r'D:\Videos\racelog_test\test6.mp4']
    for i in range(rounds):
        for v in vids:
            vr = VideoReader(v, ctx=hybrid(0), output_format='yuv420')
            vr.seek(0)
            vr.get_batch(list(range(300)))
            del vr
        print(f'  stress round {i + 1}/{rounds} done (观察 nvidia-smi 显存水位)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video')
    ap.add_argument('--n', type=int, default=int(os.environ.get('HYB_TEST_N', '600')))
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--stress', type=int, default=0)
    a = ap.parse_args()
    if a.stress:
        stress(a.stress)
    elif a.video:
        print('verify:', verify(a.video, a.n))
        if a.bench:
            bench(a.video, a.n)
    else:
        ok = True
        for v in (r'D:\Videos\racelog_test\test.mp4',
                  r'D:\Videos\racelog_test\test3.mp4',
                  r'D:\Videos\racelog_test\test6.mp4'):
            ok &= verify(v, a.n)
        print('ALL PASS' if ok else 'FAIL')
