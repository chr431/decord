"""hybrid->gpu 会话快速交替压测:直接检验"hybrid teardown 污染下一个 NVDEC 会话"假设。

用法: python tools/av1_ctx_stress.py [轮数=20]
每轮: hevc hybrid_gpu 会话(建+用+毁) -> av1 gpu 会话(建+用+毁)。
若 native 侧 ExitProcess/崩溃,外层捕获退出码;faulthandler 兜底打印栈。
"""
import faulthandler
import sys

faulthandler.enable()

from decord import VideoReader, gpu, hybrid_gpu

HEVC = r"D:\Videos\racelog_test\test.mp4"
AV1 = r"D:\Videos\racelog_test\test6.mp4"
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20

for i in range(iters):
    vr = VideoReader(HEVC, ctx=hybrid_gpu(0), output_format="gray")
    vr.get_batch(list(range(8))).asnumpy()
    del vr
    vr = VideoReader(AV1, ctx=gpu(0), output_format="gray")
    vr.get_batch(list(range(8))).asnumpy()
    del vr
    print(f"iter {i} ok", flush=True)
print("ALL-OK", flush=True)
