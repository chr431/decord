# -*- coding: utf-8 -*-
"""hybrid_gpu（device 101）验证：与纯 gpu 路径逐字节比对 + 设备指针通路。

引擎契约：get_batch 返回 CUDA 批 NDArray（yuv420 = (B, rows, w) packed
NV12；gray = (B, h, w, 1)），_ndarray_device_ptr 直接取设备指针进 kernel。
"""
import hashlib
import os
import sys

os.environ.setdefault('DECORD_LIBRARY_PATH',
                      'D:/Repo/decord/build-cuda13/Release')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from decord import VideoReader, gpu, hybrid_gpu  # noqa: E402

_VIDEO_DIR = 'D:/Videos/racelog_test'
if __name__ == '__main__' and not os.path.isdir(_VIDEO_DIR):
    sys.exit('skip: test videos not available on this machine: ' + _VIDEO_DIR)
VIDS = [r'D:\Videos\racelog_test\test.mp4',
        r'D:\Videos\racelog_test\test3.mp4',
        r'D:\Videos\racelog_test\test6.mp4']
CODECS = ['hevc', 'h264', 'av1']
N = int(os.environ.get('HYB_TEST_N', '600'))
BLOCK = 250
ROI = (100, 200, 1799, 699)   # decord 闭区间


def frames_equal(a, b):
    # byte-identity 等价判定:np.array_equal 直比,免去 tobytes 拷贝 + md5
    return a.shape == b.shape and bool((a == b).all())


def dev_ptr(batch):
    """引擎 _ndarray_device_ptr 同款：DLPack capsule -> DLTensor 设备信息。"""
    import ctypes
    cap = batch.to_dlpack()
    _get = ctypes.pythonapi.PyCapsule_GetPointer
    _get.restype = ctypes.c_void_p
    _get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = _get(cap, b"dltensor")

    class _DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]

    class _DLDataType(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                    ("lanes", ctypes.c_uint16)]

    class _DLTensor(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice),
                    ("ndim", ctypes.c_int32), ("dtype", _DLDataType),
                    ("shape", ctypes.POINTER(ctypes.c_int64))]

    t = ctypes.cast(ptr, ctypes.POINTER(_DLTensor)).contents
    shape = tuple(int(t.shape[i]) for i in range(t.ndim))
    return t.device.device_type == 2, t.device.device_id, int(t.data) != 0, shape


def compare(vr_a, vr_b, n, roi=None, seek_at=None):
    kw = {'roi': roi} if roi else {}
    total = min(n, len(vr_a))
    vr_a.seek(0); vr_b.seek(0)
    bad, matched = -1, 0
    for s in range(0, total, BLOCK):
        e = min(s + BLOCK, total)
        idx = list(range(s, e))
        ba = vr_a.get_batch(idx, **kw)
        bb = vr_b.get_batch(idx, **kw)
        if ba.shape != bb.shape:
            return f'SHAPE {ba.shape} vs {bb.shape}'
        na, nb = ba.asnumpy(), bb.asnumpy()
        for i in range(e - s):
            if frames_equal(na[i], nb[i]):
                matched += 1
            elif bad < 0:
                bad = s + i
        del ba, bb, na, nb
    ok_seek = True
    if seek_at is not None:
        vr_a.seek_accurate(seek_at); vr_b.seek_accurate(seek_at)
        ok_seek = frames_equal(vr_a[seek_at].asnumpy(), vr_b[seek_at].asnumpy())
    if bad < 0 and ok_seek:
        return None
    return f'{matched}/{total} 一致, 首错 {bad}, seek {"对" if ok_seek else "错"}'


ok = True
for i, v in enumerate(VIDS):
    for fmt in ('yuv420', 'gray'):
        vg = VideoReader(v, ctx=gpu(0), output_format=fmt)
        vh = VideoReader(v, ctx=hybrid_gpu(0), output_format=fmt)
        r = compare(vg, vh, N, seek_at=1000)
        print(f'[{CODECS[i]}] {fmt:7s} hybrid_gpu vs gpu: '
              f'{"PASS" if r is None else "FAIL: " + r}', flush=True)
        ok &= r is None
        # 设备指针通路（引擎用法）
        vh.seek(0)
        b = vh.get_batch(list(range(0, 32)))
        on_gpu, dev_id, has_ptr, shp = dev_ptr(b)
        print(f'[{CODECS[i]}] {fmt:7s} batch device: '
              f'{"CUDA" if on_gpu else "NOT-CUDA"} id={dev_id} ptr={has_ptr} '
              f'shape={shp}', flush=True)
        ok &= on_gpu and has_ptr
        del vg, vh
    # ROI
    for fmt in ('yuv420', 'gray'):
        vg = VideoReader(v, ctx=gpu(0), output_format=fmt, roi=ROI)
        vh = VideoReader(v, ctx=hybrid_gpu(0), output_format=fmt, roi=ROI)
        r = compare(vg, vh, 600, roi=ROI, seek_at=500)
        print(f'[{CODECS[i]}] {fmt:7s} ROI  hybrid_gpu vs gpu: '
              f'{"PASS" if r is None else "FAIL: " + r}', flush=True)
        ok &= r is None
        del vg, vh
print('ALL PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
