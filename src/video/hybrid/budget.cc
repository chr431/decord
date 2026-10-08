// hybrid/budget.cc —— HybridThreadedDecoder 实现 TU（R3-3 拆分，
// 2026-09-28 自 hybrid_threaded_decoder.cc 机械搬运；公共语义见
// ../hybrid_threaded_decoder.h，职责：速率/预算/侧选择/期望帧数（budget））
#include "../hybrid_threaded_decoder.h"

#include "../ffmpeg/ffmpeg_common.h"
#include <decord/runtime/ndarray.h>

#include "../nvcodec/cuda_threaded_decoder.h"

#include <dmlc/logging.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "../../runtime/cuda/cudart_shim.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstring>
#include <vector>

namespace decord {

void HybridThreadedDecoder::ComputeBudgets() {
    // adaptive: free VRAM/RAM -> bounded budgets; alloc failure = backpressure
    // 库存份额 0.45（2026-09-08 从 0.30 上调）：块交替调度下，两侧库存帽
    // （CPU queue / GPU ready·池）必须 ≥ 相位帧量，否则值日方的生产在对
    // 方值日里被背压闸住（产能闲置）—— hevc 显存池 2132→4500MB 实测
    // 引擎口径 1848→2041（每轮 >2000，超纯 NVDEC 9%）。仍按空闲量自适应，
    // env 可覆盖。
    vram_budget_ = 768.0 * 1024 * 1024;
    double ram_budget = 1536.0 * 1024 * 1024;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b > 0)
        // VRAM 0.65：VRAM 模式的 GPU 库存(sess_.ready_)持有池缓冲、与 NVDEC
        // 喂包共用一个池，池深 = 库存上限 —— av1 引擎口径实测预算
        // 3198→5200MB 时 1529→1805(GPU 主导码流需要最深库存；
        // hevc/h264 已饱和不受影响)。仍按空闲量自适应，留 35% 余量。
        vram_budget_ = static_cast<double>(free_b) * 0.65;
#if defined(_WIN32)
    { MEMORYSTATUSEX ms{}; ms.dwLength = sizeof(ms);
      if (GlobalMemoryStatusEx(&ms)) ram_budget = static_cast<double>(ms.ullAvailPhys) * 0.45; }
#else
    { long pages = sysconf(_SC_AVPHYS_PAGES); long ps = sysconf(_SC_PAGE_SIZE);
      if (pages > 0 && ps > 0) ram_budget = static_cast<double>(pages) * ps * 0.45; }
#endif
    ram_budget_ = ram_budget;
    if (const char *e = getenv("DECORD_HYBRID_VRAM_BUDGET_MB"))
        if (atof(e) > 0) vram_budget_ = atof(e) * 1024 * 1024;
    if (const char *e = getenv("DECORD_HYBRID_RAM_BUDGET_MB"))
        if (atof(e) > 0) ram_budget = atof(e) * 1024 * 1024;
    const double fb = frame_bytes_ > 0 ? static_cast<double>(frame_bytes_) : 3.1e6;
    auto clampi = [](double v, int lo, int hi) {
        return static_cast<int>(std::max<double>(lo, std::min<double>(v, hi))); };
    if (out_cuda_) {
        gpu_pool_frames_ = clampi(vram_budget_ * 0.65 / fb, 96, 1536);
        up_pool_frames_ = clampi(vram_budget_ * 0.35 / fb, 48, 768);
    } else {
        gpu_pool_frames_ = clampi(vram_budget_ * 0.65 / fb, 28, 128);
        up_pool_frames_ = 0;
    }
    // queue 钳制 768→1536（库存帽 ≥ 相位帧量的前提；RAM 预算不足时仍被
    // 预算压低，钳制只是不再先于预算生效）
    queue_frames_ = clampi(ram_budget * 0.45 / fb, 96, 1536);
    ready_cap_frames_ = clampi(ram_budget * 0.45 / fb, 96, 2560);
    if (out_cuda_) ready_cap_frames_ = gpu_pool_frames_ + up_pool_frames_ + 64;
    // demux 领先必须同时覆盖两路存货（CPU queue + GPU ready）+ 余量：
    // 否则两路分食领先窗口互相饿（CPU 块发射期 GPU 拿不到包只能空转，
    // av1 6000 帧实测退化到 836fps）。压缩包驻留 RAM 仅 ~17KB/帧。
    // 注意：深度本身不是混跑损耗的调节旋钮（hevc 1678/2600/3500/4800
    // 实测平坦）—— 决策质量由 NeedsPackets 的盲阶段节拍控制，见
    // SuggestPrefetchDepth/NeedsPackets 注释。
    prefetch_frames_ = clampi(queue_frames_ + ready_cap_frames_ + 128,
                              192, 3072);
    // AV1 384 领先钳制已删除（2026-09-10 预路由改造）：深 demux 领先
    // 是预路由设计的前提（GPU 侧要能持续拿到未来 chunk 的包）。原钳制
    // 防的跨侧交界竞态由既有 expected 对齐 + kick/陈旧丢弃兜底承担，
    // 并以全片逐位比对（引擎 _probe_hybrid_bitwise）作发布门禁。
    // 诊断/实验直控（绕过公式与 AV1 钳制）
    if (const char *e = getenv("DECORD_HYBRID_PREFETCH"))
        if (atoi(e) > 0) prefetch_frames_ = atoi(e);

    if (getenv("DECORD_HYBRID_DEBUG")) {
        fprintf(stderr, "[hybrid-budget] vram=%.0fMB ram=%.0fMB pool=%d up=%d queue=%d ready=%d prefetch=%d\n",
                vram_budget_ / 1048576.0, ram_budget / 1048576.0,
                gpu_pool_frames_, up_pool_frames_, queue_frames_,
                ready_cap_frames_, prefetch_frames_); }
}


int64_t HybridThreadedDecoder::FrameIndexOfPts(int64_t pts) const {
    // kf 索引查 pts 的呈现序帧号（窗口架构重做 2026-10-08）：GOP 判交门
    // 的基准。GOP 起点本身是关键帧 pts（精确命中）； tolerate 落点回退
    // 到 ≤pts 的最近关键帧（VFR 抖动防御），单调性保证门判定只依赖
    // 相对序。索引未注入（kf_pts_ 空）返回 -1 = 判交不可能（调用方按
    // 无窗退化）。
    if (kf_pts_.empty()) return -1;
    auto it = std::upper_bound(kf_pts_.begin(), kf_pts_.end(), pts);
    if (it == kf_pts_.begin()) return kf_rank_.front();
    --it;
    return kf_rank_[it - kf_pts_.begin()];
}


int64_t HybridThreadedDecoder::ExpectedFrames(int64_t start_pts,
                                               int64_t end_pts) const {
    // 在 kf 索引中查 [start,end) 的帧数 = rank(end_kf) - rank(start_kf)；
    // 末 chunk（end == INT64_MAX）= frame_count - rank(last_kf)。
    if (kf_pts_.empty()) return 0;  // 无索引：退化为 marker/EOF 收尾
    auto it = std::lower_bound(kf_pts_.begin(), kf_pts_.end(), start_pts);
    if (it == kf_pts_.end() || *it != start_pts) return 0;
    int64_t r0 = kf_rank_[it - kf_pts_.begin()];
    if (end_pts == INT64_MAX) {
        return frame_count_ - r0;
    }
    auto it2 = std::lower_bound(kf_pts_.begin(), kf_pts_.end(), end_pts);
    if (it2 == kf_pts_.end() || *it2 != end_pts) return 0;
    return kf_rank_[it2 - kf_pts_.begin()] - r0;
}


std::vector<int64_t> HybridThreadedDecoder::FrameShapeFor(int fmt, int h, int w) {
    // 与 VideoReader::FrameShape 同语义：yuv420 → (h+ceil(h/2), w) 的 2D；
    // gray → (h, w)；rgb → (h, w, 3)
    if (fmt == 2) {
        int64_t rows = h + (h + 1) / 2;
        return {rows, w};
    }
    int64_t c = fmt == 1 ? 1 : 3;
    return {h, w, c};
}


std::vector<int64_t> HybridThreadedDecoder::GpuFrameShape() const {
    return FrameShapeFor(output_format_, height_, width_);
}


std::size_t HybridThreadedDecoder::ReadyCap() const {
    if (ready_cap_frames_ > 0) return static_cast<std::size_t>(ready_cap_frames_);
    // ⚠️ 下面这段字节预算分支当前**不可达**（ready_cap_frames_ 在
    // ComputeBudgets/ROI 重建里恒被赋正值 ≥96）。保留为防御性兜底：
    // 若将来有路径把它置 0（如新预算模型），这里的换算仍然正确。
    // 2026-09-19 冻结前审计标记，未删。
    // 字节预算 → 帧数上限（随分辨率自适应）。下限必须 > kGpuPoolBuffers：
    // FeedStep 的喂包闸门是 sess_.ready_ 余量 ≥ 池大小，下限过小会让 GPU 永远
    // 吃不到包（活锁）。GPU 驻留模式预算翻倍（帧驻留显存直到按序消费，
    // NVDEC 超前需覆盖一个 CPU chunk 的发射期）。
    std::size_t budget = out_cuda_ ? kReadyMaxBytesGpu : kReadyMaxBytes;
    if (frame_bytes_ <= 0) return 44;
    return std::max<std::size_t>(
        44, static_cast<std::size_t>(budget / static_cast<std::size_t>(frame_bytes_)));
}


bool HybridThreadedDecoder::IsIdrLikeCodec() const {
    // H.264/HEVC 的关键帧即 IDR/CRA：kick 包可安全重置参考状态并冲刷
    // 重排帧，且输出帧 pts == 包 pts（stash 陈旧丢弃闭环成立）。
    return codec_id_ == AV_CODEC_ID_H264 || codec_id_ == AV_CODEC_ID_HEVC;
}


HybridThreadedDecoder::Side HybridThreadedDecoder::ForcedSide() const {
    // DECORD_HYBRID_FORCE_SIDE=cpu|gpu：诊断/实验用的单侧强制路由。
    // 非 IDR-like 编码（AV1 等）下 force=cpu 退化为 GPU：chunk 边界不落在
    // IDR 上时纯 CPU 分片无法按序交付（保持旧行为，只是挪到这里统一表达）。
    static const Side want = [] {
        const char *e = getenv("DECORD_HYBRID_FORCE_SIDE");
        if (!e) return Side(-1);
        return (strcmp(e, "gpu") == 0) ? SIDE_GPU : SIDE_CPU;
    }();
    if (want == SIDE_GPU) return SIDE_GPU;
    if (want == SIDE_CPU) return IsIdrLikeCodec() ? SIDE_CPU : SIDE_GPU;
    return Side(-1);
}


HybridThreadedDecoder::Side HybridThreadedDecoder::PickFeedSide() {
    // 供水式贪心（分配时点，速率=当下 sustained/EWMA 读数）：
    //   ac/(ac+ag) <= f ⟺ ac*(1-f) <= ag*f   （f = rc/(rc+rg)，env 可覆盖）
    // 启动启发：gop0 GPU（NVDEC 平价起步）；gop1 CPU 采样（IDR 码）——
    // rc 只有喂过 CPU 块才学得到；采样块与 gop0 的 GPU 解码重叠。
    // 速率未熟（采样块在解）：交替保守起点。
    // 尾部（EOF 且未供 GOP ≤4）：最少积压侧，两侧一同收尾。
    {
        const Side f = ForcedSide();
        if (f != Side(-1)) return f;
    }
    // AV1（非 IDR 语义）：CPU 混跑保留（默认开）。⚠️ 2026-09-14 二次实测
    // 否决了"GPU-only 修回归"的尝试：GPU-only = 15.1s（纯 NVDEC 水平，
    // +54% 大回归）——main 的 av1 快**正因为** CPU 混跑（引擎 10.6 vs 纯
    // NVDEC 15.2）；+2.2% 小回归另有原因（份额/切换模式差异，未归因）。
    // DECORD_HYBRID_AV1_CPU=0 可关混跑（复评用）。
    // ⚠️ 2026-09-19 冻结前修正：本开关此前**只存在于注释**（代码里没有
    // getenv），照注释设 env 会静默无效、让人误判"混跑不可关"。现补实。
    static const bool av1_cpu_off = [] {
        const char *e = getenv("DECORD_HYBRID_AV1_CPU");
        return e != nullptr && atoi(e) == 0;
    }();
    if (av1_cpu_off && !IsIdrLikeCodec()) {
        return SIDE_GPU;   // 消融：AV1 纯 GPU（复评对照臂）
    }
    if (sess_.eof_cache_ && sess_.gop_seq_ - sess_.feed_gop_idx_ <= 4) {
        return sess_.side_pending_[SIDE_CPU] <= sess_.side_pending_[SIDE_GPU]
                   ? SIDE_CPU : SIDE_GPU;
    }
    if (sess_.chunks_assigned_[SIDE_CPU] == 0) {
        return sess_.chunks_assigned_[SIDE_GPU] >= 1 ? SIDE_CPU : SIDE_GPU;
    }
    // 窗口尾收口（2026-09-18 启动轮续；2026-10-08 区间版）：EOF 的
    // 「末段给最少积压侧」规则原本只在 sess_.eof_cache_ 触发——窗口运行
    // 没有尾意识，末段 GOP 按供水比例可能落到慢臂，窗口尾部被拖满额
    // GOP 时间（hevc w1500/w2000 交叉点卡噪声带的主嫌疑）。窗口剩余
    // ≤ ~4 GOP 时切 ETA 最短侧，两侧一同收尾。剩余 = hi −（首个未派
    // GOP 的起始帧号）（区间版口径，取代旧 window−assigned 计数差）。
    // 全片（未设窗）零影响；gop0/gop1 起步启发在前，采样语义不变。
    {
        int64_t gop_est = est_chunk_frames_;
        if (gop_est <= 0 && kf_pts_.size() > 1 && frame_count_ > 0) {
            gop_est = std::max<int64_t>(
                frame_count_ / static_cast<int64_t>(kf_pts_.size() - 1), 1);
        }
        int64_t window_remaining = INT64_MAX;
        if (window_hi_ > window_lo_ && gop_est > 0) {
            for (size_t wi = static_cast<size_t>(
                     std::max<int64_t>(sess_.feed_gop_idx_, 0));
                 wi < sess_.gops_.size(); ++wi) {
                if (sess_.gops_[wi].side != Side(-1)) continue;
                const int64_t ws =
                    FrameIndexOfPts(sess_.gops_[wi].start_pts);
                // kf 缺失（ws<0）：判交不可能，尾规则退化为不触发
                //（与派工门同退化路径——宁按无窗调度）。
                if (ws >= 0) {
                    window_remaining =
                        std::max<int64_t>(window_hi_ - ws, 0);
                }
                break;
            }
        }
        if (window_remaining <= 4 * gop_est) {
            // ETA 口径（首版 least-pending 实测大回归 1.6-1.7s：
            // sess_.side_pending_ 含按序合并的等待发射帧——GPU 臂解码完成但
            // 排在 CPU chunk 后等发射时 pending 虚高，尾被系统性派给
            // 慢的 CPU 臂）。ETA=未解码余量/实测速率：pending 扣除
            // 已解码存货（CPU=filter 队列深，GPU=sess_.ready_/sess_.cpu_ready_），
            // 谁先腾手给谁；速率未熟退回供水比例（下方 water-fill）。
            const double r_c = cpu_.ProductionRate();
            const double r_g = gpu_rate_landed_.load(
                std::memory_order_relaxed);
            if (r_c > 0.0 && r_g > 0.0) {
                std::size_t gpu_inv = 0;
                {
                    std::lock_guard<std::mutex> lk(rmtx_);
                    gpu_inv = sess_.ready_.size()
                        + (out_cuda_ ? sess_.cpu_ready_.size() : 0);
                }
                // 已解码存货另含 cuvid reorder 输出队列（LandStep 尚未
                // 收割；co≤0 时诊断口径未启用，跳过）。宿主包队列与
                // cuvid parser/解码在途属未解码工作，不扣——首版把
                // 它们误计入存货（低估 GPU 余量），已更正。
                if (gpu_) {
                    int64_t cp = 0, cb = 0, co = 0;
                    gpu_->DiagDepths(&cp, &cb, &co);
                    (void)cp; (void)cb;
                    if (co > 0) gpu_inv += static_cast<std::size_t>(co);
                }
                const int64_t cpu_und = std::max<int64_t>(
                    sess_.side_pending_[SIDE_CPU] - cpu_.QueueDepth(), 0);
                const int64_t gpu_und = std::max<int64_t>(
                    sess_.side_pending_[SIDE_GPU]
                        - static_cast<int64_t>(gpu_inv), 0);
                return (static_cast<double>(cpu_und) / r_c
                        <= static_cast<double>(gpu_und) / r_g)
                           ? SIDE_CPU : SIDE_GPU;
            }
        }
    }
    const double rc = cpu_.ProductionRate();
    const double rg = gpu_rate_landed_.load(std::memory_order_relaxed);
    if (rc <= 0.0 || rg <= 0.0) {
        return sess_.chunks_assigned_[SIDE_CPU] <= sess_.chunks_assigned_[SIDE_GPU]
                   ? SIDE_CPU : SIDE_GPU;
    }
    if (!sess_.sched_initialized_) {
        // 首次双侧速率就绪：cpu-out 的库存帽按产率比分账（旧 inv-split
        // 语义不变；gpu-out 的 ready 由显存池决定不可动）
        sess_.sched_initialized_ = true;
        if (!out_cuda_) {
            const int64_t total_inv =
                static_cast<int64_t>(queue_frames_) + ready_cap_frames_;
            const double rr = rc / std::max(rc + rg, 1.0);
            int64_t q = std::max<int64_t>(
                static_cast<int64_t>(total_inv * rr), 96);
            int64_t rd = std::max<int64_t>(total_inv - q, 96);
            queue_frames_ = static_cast<int>(q);
            ready_cap_frames_ = static_cast<int>(rd);
            cpu_.SetQueueDepth(queue_frames_);
            if (pinned_pool_ && !out_cuda_) {
                pinned_pool_->Reset(
                    static_cast<std::size_t>(rd) + kD2HRingSlots + 16,
                    pinned_pool_->frame_bytes(), gpu_frame_shape_);
            }
        }
    }
    static const double fs_env = [] {
        const char *e = getenv("DECORD_HYBRID_FORCE_SHARE");
        if (!e) return -1.0;
        double v = atof(e);
        return (v > 0.0 && v < 1.0) ? v : -1.0;
    }();
    double f = fs_env > 0 ? fs_env : rc / std::max(rc + rg, 1.0);
    f = std::min(std::max(f, 0.0), 1.0);
    if (trace_on_) {
        // 穿透轮：速率就绪后的每个派工决策记 [t_us, share_bp, rc, rg]
        // ——泵的份额适应过程（启动学习→稳态）自此有轨迹。
        const size_t ti = trace_n_.load(std::memory_order_relaxed);
        if (ti < 1024) {
            const int64_t t0 = stats_t0_us_.load(std::memory_order_relaxed);
            const auto now_us = std::chrono::duration_cast<
                std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            trace_ring_[ti * 4] = t0 > 0 ? (double)(now_us - t0) : 0.0;
            trace_ring_[ti * 4 + 1] = f * 10000.0;
            trace_ring_[ti * 4 + 2] = rc;
            trace_ring_[ti * 4 + 3] = rg;
            trace_n_.store(ti + 1, std::memory_order_relaxed);
        }
    }
    return (static_cast<double>(sess_.assigned_frames_[SIDE_CPU]) * (1.0 - f)
            <= static_cast<double>(sess_.assigned_frames_[SIDE_GPU]) * f)
               ? SIDE_CPU : SIDE_GPU;
}

ffmpeg::AVPacketPtr HybridThreadedDecoder::CloneCachePacket(int64_t seq) {
    // 仅 Push/泵线程（持 mtx_ 的调用方）使用
    if (seq < sess_.cache_base_seq_ || seq >= sess_.cache_seq_) return nullptr;
    const auto &src = sess_.pkt_cache_[seq - sess_.cache_base_seq_];
    if (!src) return nullptr;
    AVPacket *c = av_packet_clone(src.get());
    if (!c) return nullptr;
    return ffmpeg::AVPacketPtr(c, [](AVPacket *p) { av_packet_free(&p); });
}


}  // namespace decord
