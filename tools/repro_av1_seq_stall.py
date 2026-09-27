# -*- coding: utf-8 -*-
"""stream 套件全相位停滞复现器 v4（2026-09-28 R3-1 取证，零扰动）。

失败形态（stream 套件 ref 段）：EOF 重试耗尽 FATAL（video_reader.cc:917），
默认预算 10240×1ms≈10.4s；带 HYBRID_DEBUG 则 4/4 不复现（stderr 打印
拖慢消费者，时序竞态被避开——Heisenbug）。单 reader 复现器 3/3 过——
失败需要套件前置的 6 个 reader（hevc/h264/av1 × gpu/hybrid）留下的
NVDEC 会话状态。

本版：完整复刻套件序列 + 放大重试预算（FATAL→慢）+ 主线程看门狗
每 5s 转储 hybrid_stats（零扰动：不动消费线程节奏）。

用法: timeout 240 python tools/repro_av1_seq_stall.py
"""
import json
import os
import sys
import threading
import time

os.environ.setdefault('DECORD_LIBRARY_PATH', 'D:/Repo/decord/build-081fix')
os.environ['DECORD_EOF_RETRY_MAX'] = '99999999'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from decord import VideoReader, gpu, hybrid, hybrid_gpu  # noqa: E402

VIDS = [(r'D:\Videos\racelog_test\test.mp4', 'hevc'),
        (r'D:\Videos\racelog_test\test3.mp4', 'h264'),
        (r'D:\Videos\racelog_test\test6.mp4', 'av1')]
N = 2000
B = 250

progress = {'phase': '', 'frames': 0, 'tp': time.time(), 'done': False,
            'vr': None}
DUMPS = []


def seq_read(v, ctx):
    """套件 ref 段的顺序 get_batch（失败原位点）。"""
    vr = VideoReader(v, ctx=ctx, output_format='yuv420')
    progress['vr'] = vr
    vr.seek(0)
    for s in range(0, N, B):
        vr.get_batch(list(range(s, min(s + B, N))))
        progress['frames'] = min(s + B, N)
        progress['tp'] = time.time()
    del vr
    progress['vr'] = None


def reader():
    # ── 与 test_hybrid_stream.py 逐语句等价（ref + stream 全相位）──
    import hashlib
    for v, codec in VIDS:
        for name, ctx in (('gpu', gpu(0)), ('hybrid', hybrid_gpu(0))):
            progress['phase'] = f'{codec}:{name}:ref'
            progress['frames'] = 0
            progress['tp'] = time.time()
            vr = VideoReader(v, ctx=ctx, output_format='yuv420')
            progress['vr'] = vr
            vr.seek(0)
            ref = []
            for s in range(0, N, B):
                ref.extend(x for x in vr.get_batch(
                    list(range(s, min(s + B, N)))).asnumpy())
                progress['frames'] = min(s + B, N)
                progress['tp'] = time.time()
            del vr
            progress['vr'] = None
            progress['phase'] = f'{codec}:{name}:stream'
            progress['frames'] = 0
            progress['tp'] = time.time()
            vr = VideoReader(v, ctx=ctx, output_format='yuv420')
            progress['vr'] = vr
            for start, batch in vr.get_batch_stream(list(range(N)), batch=B):
                arr = batch.asnumpy()
                progress['frames'] = start + arr.shape[0]
                progress['tp'] = time.time()
            del vr
            progress['vr'] = None
    progress['done'] = True


t = threading.Thread(target=reader, daemon=True)
t0 = time.time()
t.start()
last_dump = 0.0
while not progress['done'] and time.time() - t0 < 220:
    time.sleep(0.5)
    stall = time.time() - progress['tp']
    if stall >= 5 and time.time() - last_dump >= 5:
        last_dump = time.time()
        try:
            st = progress['vr'].hybrid_stats()
        except Exception as e:  # noqa: BLE001
            st = {'err': repr(e)}
        DUMPS.append({'at_s': round(time.time() - t0, 1),
                      'phase': progress['phase'],
                      'frames': progress['frames'],
                      'stall_s': round(stall, 1), 'hybrid': st})
        print(json.dumps(DUMPS[-1], default=str), flush=True)

if progress['done']:
    print(f'RESULT: PASS dumps={len(DUMPS)}')
    sys.exit(0)
print(f'RESULT: STUCK phase={progress["phase"]} frames={progress["frames"]} '
      f'dumps={len(DUMPS)}')
sys.exit(2)
