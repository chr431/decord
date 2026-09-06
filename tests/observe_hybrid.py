# -*- coding: utf-8 -*-
"""单视频单次观察: hybrid 解码 + GPU 利用率/显存每秒采样, 带硬超时.

用法: python tests/observe_hybrid.py [video] [n_frames] [timeout_s] [num_threads]
输出: 解码 fps + GPU util/VRAM min/avg/max + 逐秒明细
"""
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault('DECORD_LIBRARY_PATH',
                      r'D:/Repo/decord/build-cuda13/Release')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from decord import VideoReader, hybrid, hybrid_gpu  # noqa: E402


def sample_loop(stop, samples, t0):
    while not stop.is_set():
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2).stdout.strip()
            u, m = out.split('\n')[0].split(',')
            samples.append((time.perf_counter() - t0, int(u), int(m)))
        except Exception:
            pass
        stop.wait(1.0)


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else r'D:\Videos\racelog_test\test.mp4'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4718
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    nt = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    ctx = hybrid_gpu(0) if os.environ.get('OBS_GPU_RESIDENT') else hybrid(0)
    vr = VideoReader(video, ctx=ctx, output_format='yuv420', num_threads=nt)
    total = min(n, len(vr))
    print(f'observing {os.path.basename(video)}: {total} frames, nt={nt}, timeout={timeout}s')
    vr.seek(0)

    stop = threading.Event()
    samples = []
    t0 = time.perf_counter()
    th = threading.Thread(target=sample_loop, args=(stop, samples, t0), daemon=True)
    th.start()

    got = 0
    block = 250
    timed_out = False
    while got < total:
        if time.perf_counter() - t0 > timeout:
            timed_out = True
            break
        e = min(got + block, total)
        try:
            b = vr.get_batch(list(range(got, e)))
        except Exception as ex:
            print(f'EXCEPTION at frame {got}: {type(ex).__name__} {str(ex)[:80]}')
            break
        got += b.shape[0]
        el = time.perf_counter() - t0
        print(f'  t={el:5.1f}s frames={got}/{total} fps={got/el:.0f}', flush=True)

    stop.set()
    el = time.perf_counter() - t0
    fps = got / el if el > 0 else 0
    status = 'TIMEOUT' if timed_out else 'DONE'
    print(f'{status}: {got}/{total} frames in {el:.1f}s = {fps:.0f} fps')
    if samples:
        utils = [s[1] for s in samples]
        mems = [s[2] for s in samples]
        print(f'GPU util min/avg/max = {min(utils)}/{sum(utils)//len(utils)}/{max(utils)}%')
        print(f'VRAM  min/avg/max = {min(mems)}/{sum(mems)//len(mems)}/{max(mems)} MiB')
        for t, u, m in samples:
            print(f'  t={t:5.1f}s util={u:3d}% vram={m} MiB')


if __name__ == '__main__':
    main()
