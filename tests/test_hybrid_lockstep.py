# -*- coding: utf-8 -*-
"""hybrid 混跑字节级对照:交错消费(每批 hybrid get_batch 后逐帧 CPU next)。

与 test_hybrid.py 的 md5 口径互补:lockstep 的慢速交错消费会放大块边界
状态机的竞态(GPU 领先死锁/路由腐坏曾在该模式下最先暴露),且逐字节
比对不依赖哈希口径。3 码流 x hybrid/hybrid_gpu。
用法: python tests/test_hybrid_lockstep.py [N]
"""
import os
import sys

os.environ.setdefault('DECORD_LIBRARY_PATH',
                      'D:/Repo/decord/build-cuda13/Release')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

_VIDEO_DIR = 'D:/Videos/racelog_test'
if __name__ == '__main__' and not os.path.isdir(_VIDEO_DIR):
    sys.exit('skip: test videos not available on this machine: ' + _VIDEO_DIR)
VIDS = [('hevc', r'D:\Videos\racelog_test\test.mp4'),
        ('h264', r'D:\Videos\racelog_test\test3.mp4'),
        ('av1', r'D:\Videos\racelog_test\test6.mp4')]

import warnings  # noqa: E402
warnings.filterwarnings('ignore')
import numpy as np  # noqa: E402
import decord  # noqa: E402


def lockstep_check(path, hyb_ctx, total, bsize=250):
    """交错消费:hybrid 每批 250 帧,期间 CPU 参照逐帧 next。

    注意 next() 序号:第 N 次 next 返回帧 N-1(0 起)。
    返回首错帧号,-1 = 全对。
    """
    vr = decord.VideoReader(path, ctx=hyb_ctx, output_format='yuv420')
    ref = decord.VideoReader(path, ctx=decord.cpu(0),
                             output_format='yuv420', num_threads=4)
    got = 0
    while got < total:
        n = min(bsize, total - got)
        b = vr.get_batch(list(range(got, got + n))).asnumpy()
        for i in range(n):
            rf = ref.next().asnumpy()
            if not np.array_equal(b[i], rf):
                return got + i
        got += n
    return -1


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    fails = 0
    for codec, path in VIDS:
        for mode, ctx in (('hybrid', decord.hybrid(0)),
                          ('hyb_gpu', decord.hybrid_gpu(0))):
            bad = lockstep_check(path, ctx, n)
            tag = 'PASS' if bad < 0 else 'FAIL 首错帧=%d' % bad
            print(f'[{codec}] {mode:8s} lockstep {n}帧: {tag}', flush=True)
            if bad >= 0:
                fails += 1
    print('ALL PASS' if fails == 0 else f'{fails} FAIL')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
