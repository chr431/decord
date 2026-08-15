"""Video Reader."""
from __future__ import absolute_import

import ctypes
import numpy as np

from ._ffi.base import c_array, c_str
from ._ffi.function import _init_api
from ._ffi.ndarray import DECORDContext
from .base import DECORDError
from . import ndarray as _nd
from .ndarray import cpu, gpu
from .bridge import bridge_out

VideoReaderHandle = ctypes.c_void_p


class VideoReader(object):
    """Individual video reader with convenient indexing and seeking functions.

    Parameters
    ----------
    uri : str
        Path of video file.
    ctx : decord.Context
        The context to decode the video file, can be decord.cpu() or decord.gpu().
    width : int, default is -1
        Desired output width of the video, unchanged if `-1` is specified.
    height : int, default is -1
        Desired output height of the video, unchanged if `-1` is specified.
    num_threads : int, default is 0
        Number of decoding thread, auto if `0` is specified.
    fault_tol : int, default is -1
        The threshold of corupted and recovered frames. This is to prevent silent fault
        tolerance when for example 50% frames of a video cannot be decoded and duplicate
        frames are returned. You may find the fault tolerant feature sweet in many cases,
        but not for training models. Say `N = # recovered frames`
        If `fault_tol` < 0, nothing will happen.
        If 0 < `fault_tol` < 1.0, if N > `fault_tol * len(video)`, raise `DECORDLimitReachedError`.
        If 1 < `fault_tol`, if N > `fault_tol`, raise `DECORDLimitReachedError`.
    output_format : str, default is 'rgb'
        Output pixel format: ``'rgb'`` (HxWx3), ``'gray'`` (HxW, luma after
        color-range expansion) or ``'yuv420'`` (semi-planar NV12 packed into
        a single H*3/2 x W array: first H rows are raw luma, followed by
        ceil(H/2) rows of interleaved raw U/V 4:2:0; use
        ``get_color_range()`` to apply the same luma expansion as 'gray').
    roi : tuple of 4 ints or None
        Optional fixed half-open ROI ``(x1, y1, x2, y2)``: the decoder then
        only ever outputs that rectangle (ROI-first pipeline — CPU crops
        before color conversion, GPU converts only the ROI window).  All
        subsequent ``next_roi`` / ``get_batch(roi=...)`` calls must pass the
        same rectangle (a mismatch raises ``ValueError``); calls without a
        roi argument return the fixed rectangle.  ``None`` keeps the legacy
        behaviour: the first roi-bearing read call before any frame is
        consumed fixes the ROI automatically, later calls fall back to
        per-frame cropping.


    """
    def __init__(self, uri, ctx=cpu(0), width=-1, height=-1, num_threads=0, fault_tol=-1,
                 output_format='rgb', roi=None):
        self._handle = None
        if output_format not in ('rgb', 'gray', 'yuv420'):
            raise ValueError("output_format must be 'rgb', 'gray' or 'yuv420'")
        self._output_format = 1 if output_format == 'gray' else (2 if output_format == 'yuv420' else 0)
        assert isinstance(ctx, DECORDContext)
        fault_tol = str(fault_tol)
        if hasattr(uri, 'read'):
            ba = bytearray(uri.read())
            uri = '{} bytes'.format(len(ba))
            self._handle = _CAPI_VideoReaderGetVideoReader(
                ba, ctx.device_type, ctx.device_id, width, height, num_threads, 2, fault_tol,
                self._output_format)
        else:
            self._handle = _CAPI_VideoReaderGetVideoReader(
                uri, ctx.device_type, ctx.device_id, width, height, num_threads, 0, fault_tol,
                self._output_format)
        if self._handle is None:
            raise RuntimeError("Error reading " + uri + "...")
        self._num_frame = _CAPI_VideoReaderGetFrameCount(self._handle)
        assert self._num_frame > 0, "Invalid frame count: {}".format(self._num_frame)
        self._key_indices = None
        self._frame_pts = None
        self._avg_fps = None
        # ROI-first 解码管线状态
        self._reader_roi = None   # (x1, y1, x2, y2) 半开，None = 未固化
        self._frames_read = 0
        self._seeked = False      # seek 后内部可能已有在途解码帧
        if roi is not None:
            x1, y1, x2, y2 = (int(v) for v in roi)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    "roi 必须是有效矩形（x2>x1 且 y2>y1），当前: {}".format(roi))
            self._set_reader_roi(x1, y1, x2, y2)

    def _set_reader_roi(self, x1, y1, x2, y2):
        """固化 reader 级 ROI（ROI-first：解码器只输出该矩形）。

        必须在读取任何帧之前调用：此后解码器（CPU filter 图 / GPU 转换
        kernel + 输出池）按 ROI 尺寸工作，无法再输出其他矩形。首次带 roi
        的读帧调用会自动固化（如 RaceVideoToLog 的固定 ROI 场景）；已开始
        读帧后请求 ROI 走旧路径（每帧裁剪），保持向后兼容。
        """
        if self._frames_read > 0:
            raise RuntimeError(
                "SetRoi 必须在读取任何帧之前调用（当前已读 {} 帧）".format(
                    self._frames_read))
        _CAPI_VideoReaderSetRoi(self._handle, int(x1), int(y1), int(x2), int(y2))
        self._reader_roi = (int(x1), int(y1), int(x2), int(y2))

    def _check_roi(self, roi):
        """带 roi 的读帧调用前的 ROI 一致性处理；返回归一化 roi 元组或 None。

        reader ROI 已固化时：roi 必须一致（缺省回退到 reader ROI）；
        完全处女 reader（未读帧且未 seek）自动固化（ROI-first 快速路径）；
        已 seek / 已读帧后保持旧路径（每帧裁剪）。空矩形（x2<=x1 或
        y2<=y1）保持旧语义（回退全帧），不固化。
        """
        if roi is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in roi)
        if x2 <= x1 or y2 <= y1:
            return (x1, y1, x2, y2)  # 旧路径：全帧回退，不固化
        if self._reader_roi is not None:
            if (x1, y1, x2, y2) != self._reader_roi:
                raise ValueError(
                    "roi {} 与 reader 固化 ROI {} 不一致：ROI-first 解码器"
                    "只能输出固定矩形".format((x1, y1, x2, y2), self._reader_roi))
            return self._reader_roi
        if self._frames_read == 0 and not self._seeked:
            self._set_reader_roi(x1, y1, x2, y2)
            return self._reader_roi
        return (x1, y1, x2, y2)  # 旧路径：每帧裁剪

    def close(self):
        """Explicitly release the native reader.

        Releasing the handle deterministically lets the OS reclaim the file
        handle immediately (rename/delete on Windows, dmlc/decord#222) instead
        of waiting for GC.  Safe to call more than once; the reader cannot be
        used afterwards.
        """
        if self._handle is not None:
            _CAPI_VideoReaderFree(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except TypeError:
            pass

    def __len__(self):
        """Get length of the video. Note that sometimes FFMPEG reports inaccurate number of frames,
        we always follow what FFMPEG reports.

        Returns
        -------
        int
            The number of frames in the video file.

        """
        return self._num_frame

    def __getitem__(self, idx):
        """Get frame at `idx`.

        Parameters
        ----------
        idx : int or slice
            The frame index, can be negative which means it will index backwards,
            or slice of frame indices.

        Returns
        -------
        ndarray
            Frame of shape HxWx3 or batch of image frames with shape NxHxWx3,
            where N is the length of the slice.
        """
        if isinstance(idx, slice):
            return self.get_batch(range(*idx.indices(len(self))))
        if idx < 0:
            idx += self._num_frame
        if idx >= self._num_frame or idx < 0:
            raise IndexError("Index: {} out of bound: {}".format(idx, self._num_frame))
        self.seek_accurate(idx)
        return self.next()

    def next(self, roi=None):
        """Grab the next frame.

        Parameters
        ----------
        roi : tuple of 4 ints or None
            Optional half-open ROI ``(x1, y1, x2, y2)`` in full-frame pixel
            coordinates.  When given, only the ROI rectangle is returned
            (shape ``(y2 - y1, x2 - x1, 3)``; on the GPU path only the ROI
            is copied from device memory, avoiding a full-frame D2H copy).
            ``None`` (default) returns the full frame.

        Returns
        -------
        ndarray
            Frame with shape HxWx3 (full) or (y2-y1) x (x2-x1) x 3 (ROI).

        """
        assert self._handle is not None
        roi = self._check_roi(roi)
        if roi is not None:
            x1, y1, x2, y2 = roi
            arr = _CAPI_VideoReaderNextFrameRoi(
                self._handle, x1, y1, x2, y2)
        else:
            arr = _CAPI_VideoReaderNextFrame(self._handle)
        self._frames_read += 1
        if not arr.shape:
            raise StopIteration()
        return bridge_out(arr)

    def next_roi(self, x1, y1, x2, y2):
        """Grab the next frame, returning only the ROI rectangle.

        The ROI is half-open ``[x1, x2) x [y1, y2)`` (numpy slice semantics)
        in full-frame pixel coordinates; the result is a host-side uint8
        array of shape ``(y2 - y1, x2 - x1, 3)`` in the same RGB channel
        order as ``next()``.  On the GPU path only the ROI is copied from
        device memory, avoiding a full-frame D2H copy.  CPU builds and
        invalid/empty ROIs return the full frame instead.

        Parameters
        ----------
        x1, y1, x2, y2 : int
            Half-open ROI bounds in the full frame.

        Returns
        -------
        ndarray
            ROI crop, shape (y2 - y1, x2 - x1, 3).

        """
        assert self._handle is not None
        roi = self._check_roi((x1, y1, x2, y2))
        if roi is not None:
            x1, y1, x2, y2 = roi
            arr = _CAPI_VideoReaderNextFrameRoi(
                self._handle, int(x1), int(y1), int(x2), int(y2))
        else:
            arr = _CAPI_VideoReaderNextFrame(self._handle)
        self._frames_read += 1
        if not arr.shape:
            raise StopIteration()
        return bridge_out(arr)

    def _validate_indices(self, indices):
        """Validate int64 integers and convert negative integers to positive by backward search"""
        assert self._handle is not None
        indices = np.array(indices, dtype=np.int64)
        # process negative indices
        indices[indices < 0] += self._num_frame
        if not (indices >= 0).all():
            raise IndexError(
                'Invalid negative indices: {}'.format(indices[indices < 0] + self._num_frame))
        if not (indices < self._num_frame).all():
            raise IndexError('Out of bound indices: {}'.format(indices[indices >= self._num_frame]))
        return indices

    def get_frame_timestamp(self, idx):
        """Get frame playback timestamp in unit(second).

        Parameters
        ----------
        indices: list of integers or slice
            A list of frame indices. If negative indices detected, the indices will be indexed from backward.

        Returns
        -------
        numpy.ndarray
            numpy.ndarray of shape (N, 2), where N is the size of indices. The format is `(start_second, end_second)`.
        """
        assert self._handle is not None
        if isinstance(idx, slice):
            idx = range(*idx.indices(len(self)))
        idx = self._validate_indices(idx)
        if self._frame_pts is None:
            self._frame_pts = _CAPI_VideoReaderGetFramePTS(self._handle).asnumpy()
        return self._frame_pts[idx, :]


    def get_batch(self, indices, roi=None):
        """Get entire batch of images. `get_batch` is optimized to handle seeking internally.
        Duplicate frame indices will be optmized by copying existing frames rather than decode
        from video again.

        Parameters
        ----------
        indices : list of integers
            A list of frame indices. If negative indices detected, the indices will be indexed from backward
        roi : tuple of 4 ints or None
            Optional half-open ROI ``(x1, y1, x2, y2)``; every frame is
            cropped to the rectangle before the batch copy (batch shape
            ``(N, y2-y1, x2-x1, 3)``).  ``None`` (default) returns full
            frames, shape ``(N, H, W, 3)``.

        Returns
        -------
        ndarray
            An entire batch of image frames with shape NxHxWx3 (or
            Nx(y2-y1)x(x2-x1)x3 with roi), where N is the length of `indices`.

        """
        assert self._handle is not None
        indices = _nd.array(self._validate_indices(indices))
        roi = self._check_roi(roi)
        if roi is not None:
            x1, y1, x2, y2 = roi
            arr = _CAPI_VideoReaderGetBatchRoi(
                self._handle, indices, x1, y1, x2, y2)
        else:
            arr = _CAPI_VideoReaderGetBatch(self._handle, indices)
        self._frames_read += len(indices)
        return bridge_out(arr)

    def get_key_indices(self):
        """Get list of key frame indices.

        Returns
        -------
        list
            List of key frame indices.

        """
        if self._key_indices is None:
            self._key_indices = _CAPI_VideoReaderGetKeyIndices(self._handle).asnumpy().tolist()
        return self._key_indices

    def get_codec(self):
        """Get video codec name (e.g. h264, hevc).

        Returns
        -------
        str
            Codec name, or empty string if unavailable.

        """
        return _CAPI_VideoReaderGetCodec(self._handle)

    def get_color_range(self):
        """Get the stream luma color range.

        Returns
        -------
        int
            ``0`` = limited/tv (16-235), ``1`` = full/pc (0-255).
            ``'gray'`` output expands limited→full accordingly; ``'yuv420'``
            keeps the raw Y plane so callers can apply the same expansion.

        """
        return _CAPI_VideoReaderGetColorRange(self._handle)

    def get_avg_fps(self):
        """Get average FPS(frame per second).

        Returns
        -------
        float
            Average FPS.

        """
        if self._avg_fps is None:
            self._avg_fps = _CAPI_VideoReaderGetAverageFPS(self._handle)
        return self._avg_fps

    def seek(self, pos):
        """Fast seek to frame position, this does not guarantee accurate position.
        To obtain accurate seeking, see `accurate_seek`.

        Parameters
        ----------
        pos : integer
            Non negative seeking position.

        """
        assert self._handle is not None
        assert pos >= 0 and pos < self._num_frame
        self._seeked = True
        success = _CAPI_VideoReaderSeek(self._handle, pos)
        if not success:
            raise RuntimeError("Failed to seek to frame {}".format(pos))

    def seek_accurate(self, pos):
        """Accurately seek to frame position, this is slower than `seek`
        but guarantees accurate position.

        Parameters
        ----------
        pos : integer
            Non negative seeking position.

        """
        assert self._handle is not None
        assert pos >= 0 and pos < self._num_frame
        self._seeked = True
        success = _CAPI_VideoReaderSeekAccurate(self._handle, pos)
        if not success:
            raise RuntimeError("Failed to seek_accurate to frame {}".format(pos))

    def skip_frames(self, num=1):
        """Skip reading multiple frames. Skipped frames will still be decoded
        (required by following frames) but it can save image resize/copy operations.


        Parameters
        ----------
        num : int, default is 1
            The number of frames to be skipped.

        """
        assert self._handle is not None
        assert num > 0
        _CAPI_VideoReaderSkipFrames(self._handle, num)

_init_api("decord.video_reader")
