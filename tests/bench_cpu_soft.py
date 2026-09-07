# -*- coding: utf-8 -*-
"""纯 CPU 软解吞吐(与 bench_hybrid_gpu.py 同口径) + GOP 结构诊断."""
import os
import sys
import time

os.environ.setdefault('DECORD_LIBRARY_PATH',
                      r'D:/Repo/decord/build-cuda13/Release')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from decord import VideoReader, cpu  # noqa: E402

VIDS = [r'D:\Videos\racelog_test\test.mp4',
        r'D:\Videos\racelog_test\test3.mp4',
        r'D:\Videos\racelog_test\test6.mp4']
CODECS = ['hevc', 'h264', 'av1']
BLOCK = 250


def run(path, n):
    vr = VideoReader(path, ctx=cpu(0), output_format='yuv420', num_threads=12)
    vr.seek(0)
    t0, got = time.perf_counter(), 0
    while got < n:
        e = min(got + BLOCK, n)
        b = vr.get_batch(list(range(got, e)))
        got += b.shape[0]
        del b
    return got / (time.perf_counter() - t0), vr


def gop_info(path):
    """用 ffprobe 看 GOP 长度与编码参数."""
    import subprocess
    import json
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json',
         '-show_streams', '-select_streams', 'v:0', path],
        capture_output=True, text=True)
    s = json.loads(r.stdout)['streams'][0]
    return (f"codec={s['codec_name']} {s['width']}x{s['height']} "
            f"has_b_frames={s.get('has_b_frames')} "
            f"r_frame_rate={s.get('r_frame_rate')}")


n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
for i, v in enumerate(VIDS):
    fps, vr = run(v, n)
    print(f'[{CODECS[i]}] cpu(12t) {fps:.0f}  |  {gop_info(v)}', flush=True)
