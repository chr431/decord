"""Performance smoke test: guard against silent decode-throughput regressions.

Keeps a generous floor (10x+ headroom on CI hardware) so it only fails on
real regressions, not runner variance.
"""
import os
import time
import pytest
from decord import VideoReader, cpu

VIDEO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'examples', 'flipping_a_pancake.mkv'))

# 310 frames of 426x240 H.264.  Local dev machines measure thousands of fps
# including open; slow 2-core CI runners still reach several hundred.
MIN_SEQUENTIAL_FPS = 200.0


def test_decode_throughput_floor():
    vr = VideoReader(VIDEO, ctx=cpu(0))
    n = len(vr)
    assert n == 310
    t0 = time.perf_counter()
    for i in range(n):
        vr[i]
    elapsed = time.perf_counter() - t0
    fps = n / elapsed
    assert fps > MIN_SEQUENTIAL_FPS, \
        "sequential decode regressed: {:.0f} fps < {:.0f}".format(
            fps, MIN_SEQUENTIAL_FPS)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
