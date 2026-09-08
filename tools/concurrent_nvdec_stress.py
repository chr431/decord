"""并发 NVDEC 压力 worker:持续创建/销毁 gpu+hybrid 会话并解码,供并发复现实验用。

用法: python tools/concurrent_nvdec_stress.py <worker_id> <轮数>
"""
import faulthandler
import sys

faulthandler.enable()

from decord import VideoReader, gpu, hybrid, hybrid_gpu

WID = sys.argv[1]
iters = int(sys.argv[2])
VIDEOS = [r"D:\Videos\racelog_test\test.mp4", r"D:\Videos\racelog_test\test6.mp4"]
CTXS = [gpu, hybrid, hybrid_gpu]

for i in range(iters):
    video = VIDEOS[i % len(VIDEOS)]
    ctx = CTXS[i % len(CTXS)](0)
    vr = VideoReader(video, ctx=ctx, output_format="gray")
    vr.get_batch(list(range(10))).asnumpy()
    del vr
    print(f"[w{WID}] iter {i} ok", flush=True)
print(f"[w{WID}] ALL-OK", flush=True)
