# -*- coding: utf-8 -*-
"""混跑损耗剖析：h264 上分离 hybrid 机制开销 vs 混跑动态损耗。

配置矩阵（全部 yuv420 get_batch 口径）：
  cpu12    纯 CPU 12 线程（基准 ~1150）
  gpu      纯 NVDEC（~965）
  hyb      hybrid 默认调度（混跑）
  hyb@cpu  hybrid 引擎强制全 CPU（DECORD_HYBRID_FORCE_SIDE=cpu）→ 机制开销
  hyb@gpu  hybrid 引擎强制全 GPU → 机制开销
  hyb_gpu  hybrid_gpu（VRAM 驻留, 无 D2H）

每配置 3 轮取中位（本机 bench 方差 ±8-20%）。
用法: python tests/bench_mix.py [n]
"""
import os
import subprocess
import sys
import time

os.environ.setdefault('DECORD_LIBRARY_PATH',
                      'D:/Repo/decord/build-cuda13/Release')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

VIDEO = r'D:\Videos\racelog_test\test3.mp4'   # h264
BLOCK = 250
ROUNDS = 3


def run_one(ctx, kw, n):
    from decord import VideoReader
    vr = VideoReader(VIDEO, ctx=ctx, output_format='yuv420', **kw)
    vr.seek(0)
    t0, got = time.perf_counter(), 0
    while got < n:
        e = min(got + BLOCK, n)
        b = vr.get_batch(list(range(got, e)))
        got += b.shape[0]
        del b
    return got / (time.perf_counter() - t0)


def median(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    from decord import cpu, gpu, hybrid, hybrid_gpu
    configs = [
        ('cpu12', lambda: run_one(cpu(0), {'num_threads': 12}, n)),
        ('gpu', lambda: run_one(gpu(0), {}, n)),
        ('hyb', lambda: run_one(hybrid(0), {}, n)),
        ('hyb_gpu', lambda: run_one(hybrid_gpu(0), {}, n)),
    ]
    results = {}
    for name, fn in configs:
        xs = [fn() for _ in range(ROUNDS)]
        results[name] = median(xs)
        print(f'{name:9s} rounds={[f"{x:.0f}" for x in xs]} median={median(xs):.0f}',
              flush=True)
    # FORCE_SIDE 需要进程级 env（static local 缓存），子进程跑
    for side in ('cpu', 'gpu'):
        env = dict(os.environ, DECORD_HYBRID_FORCE_SIDE=side)
        r = subprocess.run([sys.executable, __file__, str(n), 'single_' + side],
                           env=env, capture_output=True, text=True)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
        print(f'hyb@{side:4s} (subproc) {tail}', flush=True)
        try:
            results['hyb@' + side] = float(tail.split('median=')[1])
        except Exception:
            pass
    cpu_base = results.get('cpu12', 0)
    print('\n=== h264 vs CPU 基准 ===')
    for k, v in results.items():
        print(f'{k:9s} {v:7.0f}  {v/cpu_base:.2f}x vs cpu', flush=True)


def single(side, n):
    """子进程模式：只跑 hybrid 一轮×3。"""
    from decord import hybrid
    xs = [run_one(hybrid(0), {}, n) for _ in range(ROUNDS)]
    print(f'forced={side} rounds={[f"{x:.0f}" for x in xs]} median={median(xs):.0f}',
          flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2].startswith('single_'):
        single(sys.argv[2][7:], int(sys.argv[1]))
    else:
        main()
