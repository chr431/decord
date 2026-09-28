# -*- coding: utf-8 -*-
"""解码器契约面（capability introspection，2026-09-28 R4）。

下游对 fork 能力面的假设（hybrid ctx、输出格式、硬窗、遥测键集……）
原本只能靠版本号比较或 hasattr 探测，换 DLL/换代时漂移无声。本模块把
「本构建实际提供什么」变成机器可读的一处，供**任何**下游做能力协商：

    import decord
    decord.CONTRACT_VERSION   # 契约面版本（int）
    decord.features()         # -> dict[str, bool | tuple | str]

消费方约定：
- 缺 features()/CONTRACT_VERSION 的 decord（上游原版/旧 wheel）= 无契约
  面，消费方回退自有探测逻辑，不视为错误。
- 有契约面时：CONTRACT_VERSION 超出消费方已知范围 → 消费方自行决定
  （告警或拒绝，两种都合法）；能力缺失 → 显式失败，替代深处裸崩溃。
- **契约政策**：features() 新增键 = 向后兼容；删键 / 改名 / 语义变更 =
  CONTRACT_VERSION +1（发布说明必须单列）。

键集（全部是通用解码能力，与任何具体下游无关）：
  roi_first            ROI-first（VideoReader SetRoi 构造期一次）
  next_roi_stream      next_roi 顺序流
  hybrid_ctx           hybrid ctx（宿主帧输出）
  hybrid_gpu_ctx       hybrid_gpu ctx（显存驻留输出）
  yuv420_packed_nv12   yuv420 packed NV12 输出
  gray_output          gray 单通道输出
  stride_fast          等差步长快速路径
  batch_stream         get_batch_stream chunk 流水发射
  get_color_range      get_color_range
  get_codec            get_codec
  device_ptr_layout    设备指针裸算术布局（无 pitch padding、uint8）
  hybrid_stats_keys    hybrid_stats() 键集（遥测穿透面）
  skip_loop_filter_env DECORD_SKIP_LOOP_FILTER 透传
  hard_decode_window   set_decode_window 硬窗（缺陷边界见下方键注）
"""
from __future__ import annotations

CONTRACT_VERSION = 1

# hybrid_stats() 的稳定键集（kv() 直通；新增键=向后兼容，删键/改名=
# CONTRACT_VERSION +1）。遥测消费方应只依赖此声明子集。
_HYBRID_STATS_KEYS = (
    'assigned_c', 'assigned_g', 'assigned_total',
    'busy_cpu_pkts', 'busy_cpu_us', 'busy_gpu_pics', 'busy_gpu_us',
    'cache_peak_mb', 'chunks_c', 'chunks_g', 'clones', 'force_eof',
    'frames_c', 'frames_g',
    'gpu_arm_stall',
    'hol_ev_c', 'hol_ev_g', 'hol_us_c', 'hol_us_g',
    'kicks_c', 'kicks_g', 'late', 'out_cuda', 'strag',
    'strandmax_c', 'strandmax_g',
    'up_cempty', 'up_flushes', 'up_frames', 'up_nobuf',
    'win_subs', 'window_frames',
)

FEATURES = {
    'roi_first': True,
    'next_roi_stream': True,
    'hybrid_ctx': True,
    'hybrid_gpu_ctx': True,
    'yuv420_packed_nv12': True,
    'gray_output': True,
    'stride_fast': True,
    'batch_stream': True,
    'get_color_range': True,
    'get_codec': True,
    'device_ptr_layout': 'contiguous_uint8_no_pitch',
    'hybrid_stats_keys': _HYBRID_STATS_KEYS,
    'skip_loop_filter_env': True,
    # 已知缺陷边界：硬窗可用，但「晚起点（start ≥ 窗长）+ seek_accurate +
    # 硬窗」组合存在 fork 级尾帧缺陷（窗界停喂的排空 marker 被当 EOF，
    # 尾帧由缓存容错替补——帧数守恒、尾部像素错误）。窗激活期间的替补
    # 有 latched 告警并计数进 hybrid_stats（win_subs，健康恒 0）；根治
    # 前消费方必须避开该组合（或以无窗基线做位级对照）。
    'hard_decode_window': True,
}


def features() -> dict:
    """返回当前 fork 的能力面快照（浅拷贝；键集见模块 docstring）。"""
    return dict(FEATURES)
