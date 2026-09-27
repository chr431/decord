# -*- coding: utf-8 -*-
"""解码器契约面（2026-09-28 R4 跨仓契约机器化）。

下游 video_ocr_engine 对本 fork 的全部能力假设原本散落在引擎代码注释
与 docs（tests/golden/decoder_contract.yaml DC-01..10），换 DLL/换代时
只能靠人肉比对。本模块把「fork 实际提供什么」变成机器可读的一处：

    import decord
    decord.CONTRACT_VERSION   # 契约面版本（能力键集变更时 +1）
    decord.features()         # -> dict[str, bool | tuple | str]

消费方约定（引擎侧 decode/contract.py）：
- 缺 features()/CONTRACT_VERSION 的 decord（上游原版/旧 wheel）= 无契约
  面，引擎回退结构性探测（hasattr / try-open），不视为错误。
- 有契约面时：CONTRACT_VERSION 超出引擎已知范围 → 显式告警（新 fork +
  旧引擎，可能有不兼容）；能力缺失 → 引擎按需拒绝并给出原因，不再让
  深处的崩溃裸奔。

键集与 decoder_contract.yaml 的对应：
  roi_first            DC-01  ROI-first（VideoReader SetRoi 构造期一次）
  next_roi_stream      DC-02  next_roi 顺序流（stride==1 校准路径）
  hybrid_ctx           DC-03  hybrid ctx（宿主帧输出）
  hybrid_gpu_ctx       DC-03  hybrid_gpu ctx（显存驻留输出）
  yuv420_packed_nv12   DC-04  yuv420 packed NV12 输出
  gray_output          DC-05  gray 单通道输出
  stride_fast          DC-06  等差步长快速路径（≥0.7.12）
  batch_stream         DC-07  get_batch_stream chunk 流水发射
  get_color_range      DC-08  get_color_range
  get_codec            DC-08  get_codec
  device_ptr_layout    DC-10  设备指针裸算术布局（无 pitch padding、uint8）
  hybrid_stats_keys           hybrid_stats() 键集（报告 v6 穿透面）
  skip_loop_filter_env        DECORD_SKIP_LOOP_FILTER 透传
  hard_decode_window          set_decode_window 硬窗（含晚起点缺陷边界）
"""
from __future__ import annotations

CONTRACT_VERSION = 1

# hybrid_stats() 的稳定键集（kv() 直通；新增键=向后兼容，删键/改名=
# CONTRACT_VERSION +1）。引擎报告层只消费此子集。
_HYBRID_STATS_KEYS = (
    'assigned_c', 'assigned_g', 'assigned_total',
    'busy_cpu_pkts', 'busy_cpu_us', 'busy_gpu_pics', 'busy_gpu_us',
    'cache_peak_mb', 'chunks_c', 'chunks_g', 'clones', 'force_eof',
    'frames_c', 'frames_g',
    'hol_ev_c', 'hol_ev_g', 'hol_us_c', 'hol_us_g',
    'kicks_c', 'kicks_g', 'late', 'out_cuda', 'strag',
    'strandmax_c', 'strandmax_g',
    'up_cempty', 'up_flushes', 'up_frames', 'up_nobuf',
    'window_frames',
)

FEATURES = {
    # DC-01..DC-10 见模块 docstring 映射表
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
    # C-57 边界随附：硬窗可用，但「晚起点（start≥窗长）+ seek_accurate +
    # 硬窗」组合存在 fork 级尾帧缺陷（引擎侧谓词 start<窗长 是唯一防线，
    # 见 decoder_contract.yaml 旁注与引擎 C-57）。
    'hard_decode_window': True,
}


def features() -> dict:
    """返回当前 fork 的能力面快照（浅拷贝；键集见模块 docstring）。"""
    return dict(FEATURES)
