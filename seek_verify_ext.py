#!/usr/bin/env python
"""Verify whether the CPU SeekAccurate rewind-to-0 is necessary.

Sequential-decode frames (reference) are compared pixel-by-pixel against
frames obtained via random access (SeekAccurate).  If a direct keyframe
seek (no rewind) returns the wrong frame, the random-access frame will
differ from the reference by a large margin (drift / duplicate frame).

Run twice: once with the rewind-to-0 in place (baseline, should be
diff=0) and once with it removed (to test the direct-seek hypothesis).
"""
import sys, os, random
import numpy as np
from decord import VideoReader, cpu, gpu

CTX = cpu(0)
THRESHOLD = 5  # max pixel diff that still counts as "same frame"

def verify(fn, ctx=CTX, samples=40, seed=42):
    vr = VideoReader(fn, ctx=ctx)
    n = len(vr)
    # reference: sequential decode from the start (no seeks at all)
    vr2 = VideoReader(fn, ctx=ctx)
    ref = {}
    for i in range(n):
        ref[i] = vr2.next().asnumpy()
    # random access via __getitem__ -> SeekAccurate
    random.seed(seed)
    idx = sorted(random.sample(range(n), min(samples, n)))
    bad = []
    maxdiff = 0
    for i in idx:
        vr3 = VideoReader(fn, ctx=ctx)
        f = vr3[i].asnumpy()
        d = float(np.abs(f.astype(np.int16) - ref[i].astype(np.int16)).max())
        maxdiff = max(maxdiff, d)
        if d > THRESHOLD:
            bad.append((i, d))
    # also report whether the mismatch is a shift (compare against i-1/i+1)
    shift_info = {}
    for i in bad:
        if i - 1 in ref and i + 1 in ref:
            d_prev = float(np.abs(
                VideoReader(fn, ctx=ctx)[i].asnumpy().astype(np.int16)
                - ref[i - 1].astype(np.int16)).max())
            d_next = float(np.abs(
                VideoReader(fn, ctx=ctx)[i].asnumpy().astype(np.int16)
                - ref[i + 1].astype(np.int16)).max())
            shift_info[i] = (round(d_prev, 1), round(d_next, 1))
    return n, idx, maxdiff, bad, shift_info

if __name__ == '__main__':
    videos = [a for a in sys.argv[1:]] or [
        r'D:/Repo/decord/examples/flipping_a_pancake.mkv',
        r'D:/Repo/decord/examples/count.mov',
    ]
    for fn in videos:
        print('=== %s ===' % os.path.basename(fn))
        n, idx, maxdiff, bad, shift = verify(fn)
        print('  frames=%d sampled=%d max_diff=%.0f' % (n, len(idx), maxdiff))
        if bad:
            print('  MISMATCHED %d frames:' % len(bad), bad[:8])
            for i in bad[:5]:
                print('    idx %d -> diff vs prev %.1f, vs next %.1f (shift=%s)' %
                      (i, shift[i][0], shift[i][1],
                       'YES' if shift[i][1] < shift[i][0] else 'no'))
        else:
            print('  all sampled frames match the sequential reference (rewind NOT needed)')
