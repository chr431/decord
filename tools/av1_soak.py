"""随机化 av1/NVDEC soak:扩大状态空间追捕偶发进程蒸发。

每个随机化轮次:随机 ctx/视频/输出格式/ROI/线程数/消费模式(含消费抖动、
"开而不消费直接销毁"的拆解积压路径、双 reader 重叠窗口)。每操作一行日志
(flush),进程被 native 侧杀死时,外层驱动凭退出码+日志尾部定位在飞操作,
凭 stderr 定位 FATAL 点/段错误栈。

用法: python tools/av1_soak.py [轮数=40]
良性异常(打开失败等)记录后继续;只有进程级死亡才由外层终止 soak。
"""
import faulthandler
import random
import sys
import time

faulthandler.enable()

import decord
from decord import VideoReader, cpu, gpu, hybrid, hybrid_gpu

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
VIDEOS = [
    r"D:\Repo\decord\bench_videos\synthetic_1080p60s.mp4",
    r"D:\Videos\racelog_test\test.mp4",
    r"D:\Videos\racelog_test\test6.mp4",
]
CTXS = {"cpu": cpu, "gpu": gpu, "hybrid": hybrid, "hybrid_gpu": hybrid_gpu}
CTX_W = [0.2, 0.3, 0.2, 0.3]

rng = random.Random()


def rnd_roi():
    x1 = rng.randrange(0, 900)
    y1 = rng.randrange(0, 500)
    x2 = rng.randrange(x1 + 32, 1920)
    y2 = rng.randrange(y1 + 32, 1080)
    return (x1, y1, x2, y2)


for rnd in range(ROUNDS):
    keep_prev = [] if rnd % 3 else None  # 周期性把上一轮 reader 留到本轮销毁(重叠窗口)
    nreaders = rng.choice([1, 1, 1, 2])
    try:
        readers = []
        for k in range(nreaders):
            video = rng.choice(VIDEOS)
            cname = rng.choices(list(CTXS), CTX_W)[0]
            fmt = rng.choice(["rgb", "yuv420", "gray"])
            roi = rnd_roi() if rng.random() < 0.5 else None
            nt = rng.choice([1, 2, 4, 8])
            kw = {"ctx": CTXS[cname](0), "output_format": fmt, "num_threads": nt}
            if roi:
                kw["roi"] = roi
            tag = f"R{rnd}.k{k}"
            print(f"{tag} open {video} {cname} {fmt} roi={roi} nt={nt}", flush=True)
            vr = VideoReader(video, **kw)
            n = len(vr)
            mode = rng.choice(["batch", "next", "seek", "drain_stall", "abandon"])
            if mode == "batch":
                k_n = rng.choice([1, 4, 8, 12, 24])
                idx = sorted(rng.sample(range(min(n, 120)), min(min(n, 120), k_n)))
                b = vr.get_batch(idx).asnumpy()
                print(f"{tag} batch n={len(idx)} {b.shape}", flush=True)
            elif mode == "next":
                for _ in range(rng.choice([1, 3, 6, 10])):
                    arr = vr.next().asnumpy()
                    if rng.random() < 0.4:
                        time.sleep(rng.uniform(0, 0.05))
                print(f"{tag} next ok", flush=True)
            elif mode == "seek":
                s = rng.randrange(0, max(1, min(n, 60)))
                vr.seek_accurate(s)
                hi = min(n, s + 3)
                b = vr.get_batch(list(range(s, hi))).asnumpy()
                print(f"{tag} seek {s} {b.shape}", flush=True)
            elif mode == "drain_stall":
                b = vr.get_batch(list(range(min(4, n)))).asnumpy()
                time.sleep(rng.uniform(0.05, 0.3))  # 消费者停摆,生产者积压
                b = vr.get_batch(list(range(min(4, n)))).asnumpy()
                print(f"{tag} drain_stall ok", flush=True)
            else:  # abandon:开而不用直接销毁(拆解积压帧路径)
                print(f"{tag} abandon", flush=True)
            readers.append(vr)
        del readers, keep_prev
    except Exception as e:  # 良性失败:记录继续
        print(f"R{rnd} SOFT-FAIL {type(e).__name__}: {e}", flush=True)
        readers = []
        keep_prev = []
    if rng.random() < 0.3:
        time.sleep(rng.uniform(0, 0.1))
print(f"SOAK-GEN-DONE rounds={ROUNDS}", flush=True)
