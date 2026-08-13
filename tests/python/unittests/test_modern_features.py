"""Regression tests for fork-specific API surface.

Covers the features RaceVideoToLog depends on, which upstream 0.6.0 does
not have: ``get_codec``, ``output_format='gray'``, ``next_roi`` /
``next(roi=...)`` / ``get_batch(roi=...)``, and ``NDArray.__dlpack__``.
"""
import os
import numpy as np
import pytest
from decord import VideoReader, cpu

CTX = cpu(0)

def _video_path():
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'examples', 'flipping_a_pancake.mkv'))

def _reader(**kwargs):
    return VideoReader(_video_path(), ctx=CTX, **kwargs)

def _bt601_gray(rgb):
    """BT.601 luma (what the sws gray conversion approximates)."""
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def test_get_codec():
    vr = _reader()
    assert vr.get_codec() == 'h264'
    # file-like objects work as well
    with open(_video_path(), 'rb') as f:
        vr2 = VideoReader(f, ctx=CTX)
        assert vr2.get_codec() == 'h264'


def test_output_format_gray():
    rgb = _reader()[0].asnumpy()
    gray = _reader(output_format='gray')[0].asnumpy()
    assert gray.shape == (rgb.shape[0], rgb.shape[1], 1)
    assert gray.dtype == np.uint8
    # sws fixed-point conversion approximates BT.601 luma
    assert np.abs(_bt601_gray(rgb) - gray[..., 0]).max() < 8
    # gray applies to batches and ROI as well (1 channel everywhere)
    assert _reader(output_format='gray').get_batch([0, 1]).shape == (2, 240, 426, 1)
    vr = _reader(output_format='gray')
    vr.seek_accurate(0)
    assert vr.next_roi(10, 20, 100, 60).shape == (40, 90, 1)


def test_output_format_invalid():
    with pytest.raises(ValueError):
        VideoReader(_video_path(), ctx=CTX, output_format='bgr')


def test_next_roi_crop():
    full = _reader()[0].asnumpy()
    vr = _reader()
    vr.seek_accurate(0)
    roi = vr.next_roi(10, 20, 100, 60).asnumpy()
    assert roi.shape == (40, 90, 3)
    assert np.array_equal(roi, full[20:60, 10:100])


def test_next_roi_clamps_out_of_bounds():
    full = _reader()[0].asnumpy()
    vr = _reader()
    vr.seek_accurate(0)
    # coordinates outside the frame clamp to the frame bounds
    assert np.array_equal(vr.next_roi(-5, -5, 9999, 9999).asnumpy(), full)


def test_next_roi_empty_falls_back_to_full_frame():
    full = _reader().get_batch([0, 1]).asnumpy()
    vr = _reader()
    vr.seek_accurate(0)
    assert np.array_equal(vr.next_roi(10, 10, 10, 20).asnumpy(), full[0])
    # empty ROI still consumes the frame: the next call returns frame 1
    assert np.array_equal(vr.next_roi(50, 10, 10, 20).asnumpy(), full[1])


def test_next_roi_advances_stream():
    full = _reader().get_batch([5, 6]).asnumpy()
    vr = _reader()
    vr.seek_accurate(5)
    f5 = vr.next_roi(10, 20, 100, 60).asnumpy()
    f6 = vr.next_roi(10, 20, 100, 60).asnumpy()
    assert np.array_equal(f5, full[0][20:60, 10:100])
    assert np.array_equal(f6, full[1][20:60, 10:100])


def test_next_with_roi_kwarg():
    full5 = _reader()[5].asnumpy()
    vr = _reader()
    vr.seek_accurate(5)
    roi = vr.next(roi=(10, 20, 100, 60)).asnumpy()
    assert roi.shape == (40, 90, 3)
    assert np.array_equal(roi, full5[20:60, 10:100])


def test_get_batch_roi():
    full = _reader().get_batch([0, 3]).asnumpy()
    batch = _reader().get_batch([0, 3], roi=(10, 20, 100, 60)).asnumpy()
    assert batch.shape == (2, 40, 90, 3)
    assert np.array_equal(batch[0], full[0][20:60, 10:100])
    assert np.array_equal(batch[1], full[1][20:60, 10:100])
    # duplicate indices are deduped internally, ROI applies to every entry
    dup = _reader().get_batch([0, 1, 0], roi=(10, 20, 100, 60)).asnumpy()
    assert dup.shape == (3, 40, 90, 3)
    assert np.array_equal(dup[0], dup[2])


def test_get_batch_roi_empty_falls_back_to_full_frame():
    batch = _reader().get_batch([0, 1], roi=(10, 10, 10, 20)).asnumpy()
    assert batch.shape == (2, 240, 426, 3)


def test_dlpack_device():
    vr = _reader()  # keep the reader alive: the frame aliases its pool buffer
    arr = vr[0]
    assert arr.__dlpack_device__() == (1, 0)  # kDLCpu


def test_dlpack_gpu_torch():
    """GPU frames export zero-copy to torch via DLPack (skipped without CUDA)."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    import ctypes
    from decord import gpu
    vr = VideoReader(_video_path(), ctx=gpu(0))
    arr = vr[0]
    t = torch.from_dlpack(arr)
    assert t.device.type == 'cuda'
    assert t.dtype == torch.uint8
    assert tuple(t.shape) == (240, 426, 3)
    assert torch.equal(t, torch.from_numpy(arr.asnumpy()).cuda())
    # zero-copy: the tensor aliases the NDArray's device buffer
    buf = ctypes.cast(arr.handle.contents.data, ctypes.c_void_p).value
    assert t.data_ptr() == buf
    # the tensor is usable in torch ops
    assert t.float().mean().item() > 0
    del t
    # the buffer returns to the pool via the deleter; reader keeps working
    assert vr[1].asnumpy().shape == (240, 426, 3)


def test_dlpack_zero_copy_roundtrip():
    import ctypes
    vr = _reader()  # keep the reader alive: the frame aliases its pool buffer
    arr = vr[0]
    view = np.from_dlpack(arr)
    assert view.shape == (240, 426, 3)
    assert np.array_equal(view, arr.asnumpy())
    # zero-copy: the dlpack view aliases the NDArray buffer directly
    # (numpy 2.x marks from_dlpack views read-only, so aliasing is verified
    # by address rather than a write-through)
    buf_addr = ctypes.cast(arr.handle.contents.data, ctypes.c_void_p).value
    assert view.ctypes.data == buf_addr


def test_index_cache_reopen():
    """Repeated opens reuse the on-disk frame index cache and must produce
    frame metadata identical to a fresh full scan."""
    import decord
    # prime the cache
    vr = _reader()
    n = len(vr)
    keys = vr.get_key_indices()
    ts = vr.get_frame_timestamp(range(10))
    # fresh scan (cache disabled) must agree with the cached open
    import os
    os.environ['DECORD_DISABLE_INDEX_CACHE'] = '1'
    try:
        vr_fresh = _reader()
        assert len(vr_fresh) == n
        assert vr_fresh.get_key_indices() == keys
        assert np.array_equal(vr_fresh.get_frame_timestamp(range(10)), ts)
    finally:
        del os.environ['DECORD_DISABLE_INDEX_CACHE']
    # and the cached open itself (second open in the same process)
    vr_cached = _reader()
    assert len(vr_cached) == n
    assert vr_cached.get_key_indices() == keys
    assert np.array_equal(vr_cached.get_frame_timestamp(range(10)), ts)


def test_pipeline_sequential_and_seek_interleaved():
    # exercises the two-stage decode pipeline (decode worker + filter worker)
    vr = _reader()
    n = len(vr)
    for i in range(n):
        vr[i]
    # random access after a full sequential pass
    vr2 = _reader()
    for i in [7, 3, 150, 300, 299, 0]:
        vr2.seek_accurate(i)
        vr2.next()


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
