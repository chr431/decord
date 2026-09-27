# -*- coding: utf-8 -*-
"""发布产物冒烟校验（R3-5，2026-09-28 重做；取代 release_check/ 的
0.7.0 zip 抽验——该 zip 与 extracted/ 共 238MB 零引用残留已删）。

对**现行构建产物**做发布前最小面校验：
  1. 契约面在位（CONTRACT_VERSION + features()，R4 起发布产物的硬面）
  2. 四 ctx 打开 + 解码（cpu / gpu / hybrid / hybrid_gpu）
  3. hybrid 帧 == cpu 帧逐位（gray，bit-exact 契约）
  4. get_codec / get_color_range / next_roi / hybrid_stats 键集抽查

用法:
    python tests/smoke_release.py [dll_dir] [video]
    dll_dir 默认 build-081fix（dev 构建）；video 默认 RACELOG_VIDEO_DIR
    环境变量下第一个 mp4，再退 racelog_test/test.mp4。
退出码 0 = 全绿；视频缺失时 skip（exit 0，CI 无视频环境的诚实降级）。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'python'))
_dll_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, 'build-081fix')
os.environ['DECORD_LIBRARY_PATH'] = _dll_dir

import decord  # noqa: E402
from decord import VideoReader, cpu, gpu, hybrid, hybrid_gpu  # noqa: E402

_vid = (sys.argv[2] if len(sys.argv) > 2 else
        (lambda d: os.path.join(d, sorted(f for f in os.listdir(d)
                                          if f.endswith('.mp4'))[0])
         (os.environ['RACELOG_VIDEO_DIR'])
         if os.environ.get('RACELOG_VIDEO_DIR') else
         r'D:\Videos\racelog_test\test.mp4'))

if not os.path.isfile(_vid):
    print('skip: 无测试视频（%s）——契约面检查仍执行' % _vid)
    _vid = None

N = 250
FAILS = []


def check(label, fn):
    try:
        detail = fn()
        print('[OK]  %-38s %s' % (label, detail or ''))
    except Exception as e:  # noqa: BLE001 - 冒烟脚本要全量报告
        FAILS.append(label)
        print('[FAIL] %-38s %s: %s' % (label, type(e).__name__, e))


def contract_surface():
    v = decord.CONTRACT_VERSION
    feats = decord.features()
    assert isinstance(v, int) and v >= 1
    for k in ('roi_first', 'hybrid_ctx', 'hybrid_gpu_ctx', 'yuv420_packed_nv12',
              'gray_output', 'stride_fast', 'batch_stream', 'get_color_range',
              'get_codec', 'hybrid_stats_keys', 'skip_loop_filter_env',
              'hard_decode_window'):
        assert feats.get(k), 'features 缺 %s' % k
    return 'CONTRACT_VERSION=%d, %d keys' % (v, len(feats))


def decode_n(ctx, fmt='gray'):
    def _f():
        vr = VideoReader(_vid, ctx=ctx, output_format=fmt)
        vr.seek(0)
        b = vr.get_batch(list(range(min(N, len(vr)))))
        del vr
        return '%d 帧 %s' % (b.shape[0], str(b.shape[1:]))
    return _f


def hybrid_bitexact():
    vc = VideoReader(_vid, ctx=cpu(0), output_format='gray', num_threads=12)
    vh = VideoReader(_vid, ctx=hybrid(0), output_format='gray')
    vc.seek(0); vh.seek(0)
    n = min(N, len(vc))
    bc = vc.get_batch(list(range(n))).asnumpy()
    bh = vh.get_batch(list(range(n))).asnumpy()
    del vc, vh
    assert bc.shape == bh.shape and (bc == bh).all(), 'hybrid 帧 != cpu 帧'
    return '%d 帧逐位一致' % n


def api_probes():
    vr = VideoReader(_vid, ctx=hybrid(0), output_format='yuv420')
    out = ['codec=%s' % vr.get_codec(), 'cr=%s' % vr.get_color_range()]
    vr.seek(0)
    out.append('next_roi %s' % str(vr.next_roi(0, 0, 640, 360).shape))
    st = vr.hybrid_stats()
    missing = [k for k in ('frames_c', 'frames_g', 'force_eof', 'cache_peak_mb')
               if k not in st]
    assert not missing, 'hybrid_stats 缺 %s' % missing
    out.append('stats %d 键' % len(st))
    del vr
    return ' '.join(out)


check('契约面（CONTRACT_VERSION+features）', contract_surface)
if _vid:
    check('cpu 解码', decode_n(cpu(0)))
    check('gpu 解码（NVDEC）', decode_n(gpu(0)))
    check('hybrid 解码', decode_n(hybrid(0)))
    check('hybrid_gpu 解码（yuv420）', decode_n(hybrid_gpu(0), 'yuv420'))
    check('hybrid == cpu 逐位', hybrid_bitexact)
    check('API 面（codec/cr/next_roi/stats）', api_probes)

print('SMOKE', 'ALL PASS' if not FAILS else 'FAIL: %s' % FAILS)
sys.exit(1 if FAILS else 0)
