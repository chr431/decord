"""av1+NVDEC 偶发原生崩溃复现脚本。

用法: python tools/av1_crash_repro.py <video> <iters> <seek|noseek> [ctx]
每轮 = 完整打开 reader(会话创建)→ 取 12 帧 → (可选 seek_accurate 再取批)
→ reader 销毁(会话销毁)。复现指纹脚本的真实调用序列。
faulthandler 开启:若是原生段错误,打印崩溃时 Python 栈;若静默退出/挂死,
则由外层记录退出码/超时。
"""
import faulthandler
import sys

faulthandler.enable()

import decord
from decord import VideoReader, cpu, gpu, hybrid, hybrid_gpu

video, iters, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ctx_name = sys.argv[4] if len(sys.argv) > 4 else "gpu"
ctx = {"cpu": cpu, "gpu": gpu, "hybrid": hybrid, "hybrid_gpu": hybrid_gpu}[ctx_name](0)

print(f"decord={decord.__version__} video={video} iters={iters} mode={mode} ctx={ctx_name}", flush=True)
for i in range(iters):
    vr = VideoReader(video, ctx=ctx, output_format="gray")
    b = vr.get_batch(list(range(12))).asnumpy()
    if mode == "seek":
        vr.seek_accurate(5)
        vr.get_batch([5, 6, 7]).asnumpy()
    del vr
    print(f"iter {i} ok shape={b.shape}", flush=True)
print("ALL-OK", flush=True)
