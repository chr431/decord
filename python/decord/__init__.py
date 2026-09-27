"""Decord python package"""
from . import function

from ._ffi.runtime_ctypes import TypeCode
from ._ffi.function import register_func, get_global_func, list_global_func_names, extract_ext_funcs
from ._ffi.base import DECORDError, DECORDLimitReachedError, __version__

# 解码器契约面（R4 跨仓契约机器化，2026-09-28）：下游按
# decord.features()/CONTRACT_VERSION 协商能力，取代版本号门控。
from ._contract import CONTRACT_VERSION, features  # noqa: F401

from .base import ALL

from . import ndarray as nd
from .ndarray import cpu, gpu, hybrid, hybrid_gpu
from . import bridge
from . import logging
from .video_reader import VideoReader
from .video_loader import VideoLoader
from .audio_reader import AudioReader
from .av_reader import AVReader
from .video_reader import probe, get_ffmpeg_version

# 实际加载的 FFmpeg 版本（见 get_ffmpeg_version 文档）。'unknown' = 原生库
# 早于该 API，或进程内 FFmpeg DLL 撞车且无法报告 —— 部署校验用它。
try:
    __ffmpeg_version__ = get_ffmpeg_version()
except Exception:  # pragma: no cover - FFI 层异常一律降级
    __ffmpeg_version__ = 'unknown'

logging.set_level(logging.ERROR)
