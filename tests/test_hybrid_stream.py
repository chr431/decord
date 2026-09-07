# -*- coding: utf-8 -*-
"""get_batch_stream（chunk 粒度流水发射）正确性验证。

判据：流式交付的帧与顺序 get_batch 逐字节一致；批内保序；无帧丢失/重复。
"""
import hashlib
import os
import sys

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
N = 2000
B = 250

ok = True
for v, codec in VIDS:
    for name, ctx in (('gpu', gpu(0)), ('hybrid', hybrid_gpu(0))):
        vr = VideoReader(v, ctx=ctx, output_format='yuv420')
        vr.seek(0)
        ref = []
        for s in range(0, N, B):
            ref.extend(x for x in vr.get_batch(list(range(s, min(s + B, N)))).asnumpy())
        del vr
        vr = VideoReader(v, ctx=ctx, output_format='yuv420')
        got = {}
        order = []
        for start, batch in vr.get_batch_stream(list(range(N)), batch=B):
            arr = batch.asnumpy()
            for i in range(arr.shape[0]):
                got[start + i] = hashlib.md5(arr[i].tobytes()).hexdigest()
            order.append(start)
        del vr
        match = sum(1 for i in range(N)
                    if got[i] == hashlib.md5(ref[i].tobytes()).hexdigest())
        complete = len(got) == N
        in_order = order == sorted(order)
        ok &= match == N and complete
        print(f'[{codec}] {name}: {match}/{N} 一致, 完整={complete}, '
              f'批序={ "顺序" if in_order else "完成序"}', flush=True)
print('STREAM TEST', 'ALL PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
