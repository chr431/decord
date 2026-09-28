// hybrid/supervisor.cc —— HybridThreadedDecoder 实现 TU（R3-3 拆分，
// 2026-09-28 自 hybrid_threaded_decoder.cc 机械搬运；公共语义见
// ../hybrid_threaded_decoder.h，职责：生命周期/错误/亲和/统计/reset（supervisor））
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

/*!
 *  Copyright (c) 2026 decord fork contributors
 * \file hybrid_threaded_decoder.cc
 * \brief CPU+GPU 混合线程解码器实现。
 *
 * 设计（与 hybrid_decoder_prototype 的 online 模式同源，落入 decord 架构；
 * GPU 数据通路见 hybrid_threaded_decoder.h 头注释）：
 *
 *   VideoReader (单 demux 线程, 顺序 Push packet, Push 永不阻塞)
 *        │  Push(pkt)
 *        ▼
 *   ┌─ HybridThreadedDecoder ──────────────────────────────────────┐
 *   │  路由: 关键帧包 = chunk 边界. 新 chunk 分给 "预测排空最早" 的侧   │
 *   │        (到新 chunk 起点的距离 / 速率 EWMA；GPU 待发射字节超预算  │
 *   │        则强制 CPU，防调度失衡)                                 │
 *   │   ├─► FFMPEGThreadedDecoder   (CPU 软解, 多线程, 自带背压)     │
 *   │   └─► gpu_pkt_q_ ─► GpuWorkerLoop ─► CUThreadedDecoder (NVDEC)│
 *   │                        └─► gpu_->Pop → D2H → ready_（有界）    │
 *   │  合并: emit_queue_ 按分配序记录 chunk {side, start_pts, end_pts,│
 *   │        expected}. Pop 只从队首 chunk 的侧取帧; 发射满 expected  │
 *   │        帧才关闭 chunk —— 与原型 "块长已知" 的语义对齐。          │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * 跨侧切换的冲刷（kick，与原型 "多解一帧" 同型）：
 *  重排深的码流要看到后续包才交出上一 chunk 尾部的重排帧。跨侧切换后
 *  离场侧断流，把新 chunk 的关键帧包克隆一份推给它（kick）即可触发冲刷。
 *  前提（仅 IDR 型码流成立）：kick 帧输出 pts == 包 pts，被 stash 扣住，
 *  迟到的重排帧先发，expected 补满关 chunk —— 全局顺序天然正确。
 *  AV1 的 show_existing/时间戳映射使 kick 帧输出 pts 落回上一 chunk
 *  区间（实测双发错位）——AV1 的 CPU 混跑默认保留（GPU-only 二次实测
 *  +54% 大回归，见 PickFeedSide 注释；DECORD_HYBRID_AV1_CPU=0 消融）。
 *
 * 其余要点：
 *  - Seek/Clear 先停 GPU 工作线程，再清两子解码器并重置路由状态；
 *    速率 EWMA 与 kf 索引保留。
 *  - EOF: CPU 侧转发 flush；GPU 侧由工作线程在包队列排空后补推
 *    kMaxOutputSurfaces 个 flush 缓冲。
 *  - codecpar 隔离: GPU 子解码器初始化 bsf (mp4→annexb) 时会就地改写
 *    传入的 codecpar，CPU 侧必须继续用原始 AVCC 参数 —— GPU 持独立副本。
 *  - 非 key 包按 pts 归属路由（解码序与 pts 序不一致的码流兜底）。
 *  - expected 帧数由 VideoReader 的关键帧索引提供（SetKeyframeRanks）。
 */


namespace {
/*! 调度粘性：同侧最少连续分配的 chunk 数（块状分配，防流水线冷启动） */
#if defined(_WIN32)
std::vector<unsigned long> SnapshotThreads();   // 定义见亲和分区节
bool SetThreadAffinity(unsigned long tid, uintptr_t mask);
#endif
}  // namespace


/*! 工作线程异常统一收口（2026-09-19 崩溃类修复；2026-09-20 改 per-instance
 *  成员——原进程级全局槽在多 reader 并存时任一 reader 的 worker 异常会
 *  污染所有 hybrid reader 的 Pop，引擎池 _POOL_MAX_TOTAL=16 下不可接受）：
 *  记录原消息（Pop 处转 DECORDError）+ 打 stderr 一份（即使 Pop 未被
 *  再调用也能留诊断）。 */
void HybridThreadedDecoder::TrapWorkerError(const char *who,
                                            const std::exception &e) {
    {
        std::lock_guard<std::mutex> lk(err_mtx_);
        if (!err_seen_.load()) {
            err_msg_ = std::string("hybrid ") + who + ": " + e.what();
            err_seen_.store(true);
        }
    }
    fprintf(stderr, "[hybrid-EXC] %s: %s\n", who, e.what());
    fflush(stderr);
}

HybridThreadedDecoder::HybridThreadedDecoder(int device_id,
                                             AVCodecParameters *codecpar,
                                             const AVInputFormat *iformat,
                                             bool output_cuda)
    : cpu_(), device_id_(device_id), out_cuda_(output_cuda) {
    // 包缓存字节预算（供料期派工的 demux 领先界；最小 64MB）。
    // kick_guard_ 默认 0 = 只发单个 IDR kick、不做债务克隆——旧反馈式
    // 清偿在本架构下会洪泛（分配领先交付 ⇒ 换侧债务快照=巨额 pending，
    // 每次换侧克隆数十包到离场侧，离场侧解出整段错侧 GOP → 无限陈旧帧
    // 活锁，实测 hevc 12000 帧挂死）。离场侧 DPB 尾帧由其下一次自身
    // GOP 的 IDR 自然冲刷；>0 可恢复克隆护栏（酷刑流复测用）。
    kick_guard_ = 0;
    if (const char *e = getenv("DECORD_HYBRID_KICK_BURST")) {
        kick_guard_ = atoi(e);
    }
    // 遥测穿透轮：份额轨迹默认关（首 1024 个派工决策，ring 4KB——
    // 诊断时开，A/B 热路径零成本）
    trace_on_ = getenv("DECORD_HYBRID_TRACE") != nullptr;
    if (const char *e = getenv("DECORD_HYBRID_PKT_CACHE_MB")) {
        const long long v = atoll(e);
        if (v > 0) cache_budget_ = static_cast<size_t>(v) << 20;
    }
    // GPU 子解码器初始化 bsf (mp4→annexb) 时会就地改写传入的 codecpar。
    // 传副本；CPU 侧继续用 VideoReader 手里的原始 AVCC 参数（其
    // avcodec_open2 发生在本构造之后），GPU 侧后续用被转换的副本。
    gpu_codecpar_.reset(avcodec_parameters_alloc());
    CHECK(gpu_codecpar_ != nullptr) << "avcodec_parameters_alloc failed";
    CHECK_GE(avcodec_parameters_copy(gpu_codecpar_.get(), codecpar), 0)
        << "avcodec_parameters_copy failed";
    gpu_.reset(new cuda::CUThreadedDecoder(device_id, gpu_codecpar_.get(), iformat));
}

HybridThreadedDecoder::~HybridThreadedDecoder() {
    if (getenv("DECORD_HYBRID_DEBUG")) fprintf(stderr, "[hybrid-d] enter\n");
    Stop();
    if (getenv("DECORD_HYBRID_DEBUG")) fprintf(stderr, "[hybrid-d] stopped\n");
}

// ── PinnedHostFramePool：D2H 直达最终帧的 pinned 主机帧池 ──
void PinnedHostFramePool::Reset(std::size_t max_cap, std::size_t frame_bytes,
                                std::vector<int64_t> shape) {
    std::lock_guard<std::mutex> lk(mtx_);
    // 旧尺寸块解除复用（在途块由 deleter 归还时按尺寸直接释放）
    while (!free_.empty()) {
        cudaFreeHost(free_.front());
        free_.pop_front();
    }
    max_cap_ = max_cap;
    cap_ = std::min<std::size_t>(max_cap, 64);  // 起步小池，耗尽翻倍
    bytes_ = frame_bytes;
    shape_ = std::move(shape);
}

bool PinnedHostFramePool::Acquire(runtime::NDArray *out) {
    if (disabled_ || max_cap_ == 0 || bytes_ == 0) return false;
    void *p = nullptr;
    bool need_alloc = false;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!free_.empty()) {
            p = free_.front();
            free_.pop_front();
        } else if (created_ < cap_) {
            ++created_;
            need_alloc = true;
        } else if (cap_ < max_cap_) {
            cap_ = std::min<std::size_t>(cap_ * 2, max_cap_);
            ++created_;
            need_alloc = true;
        }
        // else：池尽 → 背压（p 保持 null）
    }
    if (need_alloc) {
        if (cudaHostAlloc(&p, bytes_, cudaHostAllocDefault) != cudaSuccess) {
            // OOM/权限：禁用池，永久回退暂存路径（正确性优先）
            std::lock_guard<std::mutex> lk(mtx_);
            --created_;
            disabled_ = true;
            return false;
        }
    }
    if (p == nullptr) return false;
    Holder *h = new Holder{shared_from_this(), p, bytes_};
    *out = runtime::NDArray::FromRecycled(p, shape_, kUInt8,
                                          DLDevice{kDLCPU, 0},
                                          &ReturnDeleter, h);
    return true;
}

void PinnedHostFramePool::ReturnDeleter(runtime::NDArray::Container *self) {
    auto *h = static_cast<Holder *>(self->manager_ctx);
    h->pool->Return(h->data, h->bytes);
    delete h;
    delete self;
}

void PinnedHostFramePool::Return(void *data, std::size_t bytes) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!disabled_ && bytes == bytes_ && free_.size() < max_cap_) {
        free_.push_back(data);
    } else {
        cudaFreeHost(data);
    }
}

void HybridGpuBufferPool::Reset(
        std::size_t cap, std::vector<int64_t> shape, DLDataType dtype, DLDevice dev) {
    std::lock_guard<std::mutex> lk(mtx_);
    // 旧形状的缓冲直接解除池关联（析构走通用释放路径）
    while (!free_.empty()) {
        auto arr = free_.front();
        free_.pop_front();
        arr.data_->manager_ctx = nullptr;
    }
    max_cap_ = cap;
    cap_ = std::min<std::size_t>(cap, 64);  // 起步小池，耗尽翻倍增长
    shape_ = std::move(shape);
    dtype_ = dtype;
    dev_ = dev;
    created_ = 0;
    running_ = true;
}

HybridGpuBufferPool::~HybridGpuBufferPool() {
    std::lock_guard<std::mutex> lk(mtx_);
    running_ = false;
    while (!free_.empty()) {
        auto arr = free_.front();
        free_.pop_front();
        // 解除关联后由 NDArray 析构走通用释放（此刻池已不再被引用）
        arr.data_->manager_ctx = nullptr;
    }
}

void HybridGpuBufferPool::Start() {
    std::lock_guard<std::mutex> lk(mtx_);
    running_ = true;
    cv_.notify_all();
}

void HybridGpuBufferPool::Stop() {
    std::lock_guard<std::mutex> lk(mtx_);
    running_ = false;
    cv_.notify_all();
}

bool HybridGpuBufferPool::Acquire(runtime::NDArray *out) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!free_.empty()) {
        *out = free_.front();
        free_.pop_front();
        return true;
    }
    if (running_ && created_ >= cap_ && cap_ < max_cap_) {
        cap_ = std::min<std::size_t>(cap_ * 2, max_cap_);  // 耗尽翻倍
    }
    if (running_ && created_ < cap_) {
        // 池空且未建满：新建（总量受 cap_ 硬约束，显存边界）
        ++created_;
        auto arr = runtime::NDArray::Empty(shape_, dtype_, dev_);
        arr.data_->manager_ctx = this;
        arr.data_->deleter = &HybridGpuBufferPool::Deleter;
        *out = arr;
        return true;
    }
    // 池空且已建满：GPU 解码超前已满 —— 拒绝（不阻塞，工作线程
    // 下轮 LandStep 回收后再试）
    return false;
}

void HybridGpuBufferPool::Deleter(runtime::NDArray::Container *ptr) {
    if (!ptr) return;
    if (ptr->manager_ctx == nullptr) {
        if (ptr->dl_tensor.data != nullptr) {
            runtime::DeviceAPI::Get(ptr->dl_tensor.device)->FreeDataSpace(
                ptr->dl_tensor.device, ptr->dl_tensor.data);
        }
        delete ptr;
        return;
    }
    auto *pool = static_cast<HybridGpuBufferPool *>(ptr->manager_ctx);
    {
        std::lock_guard<std::mutex> lk(pool->mtx_);
        if (pool->free_.size() + 1 > pool->cap_) {
            // 超额（理论不可达）：直接释放
            pool->cv_.notify_all();
        } else {
            // 重新挂上池的 deleter，交回自由队列（NDArrayPool 同款手法）
            pool->free_.push_back(runtime::NDArray(ptr));
            pool->cv_.notify_all();
            if (pool->on_release_) pool->on_release_();
            return;
        }
    }
    if (ptr->dl_tensor.data != nullptr) {
        runtime::DeviceAPI::Get(ptr->dl_tensor.device)->FreeDataSpace(
            ptr->dl_tensor.device, ptr->dl_tensor.data);
    }
    delete ptr;
}


void HybridThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, int width,
                                             int height, int rotation,
                                             int output_format) {
    width_ = width;
    height_ = height;
    rotation_ = rotation;
    output_format_ = output_format;
    codec_id_ = dec_ctx->codec_id;
    ResetRouting();
    if (gpu_) {
        // GPU 输出形状已知即可算预算（frame_bytes_ 自算），所有池深/队列
        // 深度在子解码器/池初始化前就位 —— 此前预算从未被调用（池恒为
        // 默认 128 帧，GPU 超前覆盖不了 CPU chunk 发射期，慢侧弃用闸
        // 因此被引入）。
        gpu_frame_shape_ = GpuFrameShape();
        frame_bytes_ = 1;  // kUInt8
        for (int64_t d : gpu_frame_shape_) frame_bytes_ *= d;
        ComputeBudgets();
    }
    // CPU 子解码器接管 VideoReader 打开的 ctx（内部 dec_ctx_.reset 持有）
    // 深存货队列：CPU chunk 的发射靠 cpu_ready_/cpu_ 内部存货瞬时完成，
    // 默认 32 帧背压会让每个 CPU chunk 退化为实时跟随解码（hevc 0.70x）。
    cpu_.SetQueueDepth(queue_frames_);  // 存货深度与 prefetch 匹配（~1.2GB RAM@1080p）
#if defined(_WIN32)
    InitAffinityMasks();
    if (affinity_on_) aff_baseline_ = SnapshotThreads();
#endif
    cpu_.SetCodecContext(dec_ctx, width, height, rotation, output_format);
#if defined(_WIN32)
    PinFfmpegThreads();
#endif
    if (gpu_) {
        // GPU 子解码器需要自己的 AVCodecContext（CUThreadedDecoder 的
        // SetCodecContext 同样接管所有权）。用 bsf 转换后的副本参数
        // （extradata 已是 annexb，cuvid parser 需要）单独 open。
        const AVCodec *codec = avcodec_find_decoder(gpu_codecpar_->codec_id);
        CHECK(codec != nullptr) << "avcodec_find_decoder failed for hybrid GPU side";
        AVCodecContext *gctx = avcodec_alloc_context3(codec);
        CHECK(gctx != nullptr) << "avcodec_alloc_context3 failed";
        CHECK_GE(avcodec_parameters_to_context(gctx, gpu_codecpar_.get()), 0)
            << "avcodec_parameters_to_context failed (hybrid GPU side)";
        gctx->thread_count = dec_ctx->thread_count;
        gctx->time_base = dec_ctx->time_base;
        CHECK_GE(avcodec_open2(gctx, codec, nullptr), 0)
            << "avcodec_open2 failed (hybrid GPU side)";
        gpu_->SetCodecContext(gctx, width, height, rotation, output_format);
        // GPU 输出缓冲池（预算已在函数开头 ComputeBudgets 就位）
        gpu_pool_.Reset(gpu_pool_frames_,
                        gpu_frame_shape_, kUInt8,
                        DLDevice{kDLCUDA, device_id_});
        // 事件驱动：子解码器产出 / 池回收 → 即时唤醒工作线程
        gpu_->SetOnOutput([this](bool marker) {
            if (!marker) GpuFrameProduced();
            lcv_.notify_all();
        });
        cpu_.SetOnOutput([this] { lcv_.notify_all(); });
        gpu_pool_.SetOnRelease([this] { lcv_.notify_all(); });
        up_pool_.SetOnRelease([this] { lcv_.notify_all(); });
        if (out_cuda_) {
            // 上载池容量 = ready_ 预算的一半（帧数）：CPU chunk 的
            // 显存帧容器，独立于 GPU 解码池（防饿死）
            up_pool_.Reset(up_pool_frames_, gpu_frame_shape_, kUInt8,
                           DLDevice{kDLCUDA, device_id_});
        } else {
            // pinned 主机帧池（D2H 直达最终帧）：池深 = ready 预算 +
            // 在途/余量 —— pinned 只是替换 ready 帧原有的 pageable 分配
            //（内存总量不变，非分页锁定）。cudaHostAlloc 失败自动禁用
            // 回退暂存路径；DECORD_HYBRID_PINNED_POOL=0 显式关闭。
            static const bool pinned_on = [] {
                const char *e = getenv("DECORD_HYBRID_PINNED_POOL");
                return e == nullptr || atoi(e) != 0;
            }();
            if (pinned_on) {
                if (!pinned_pool_) {
                    pinned_pool_ = std::make_shared<PinnedHostFramePool>();
                }
                pinned_pool_->Reset(
                    static_cast<std::size_t>(ready_cap_frames_)
                        + kD2HRingSlots + 16,
                    static_cast<std::size_t>(frame_bytes_), gpu_frame_shape_);
            }
        }
    }
}


void HybridThreadedDecoder::SetKeyframeRanks(std::vector<int64_t> pts_list,
                                              std::vector<int64_t> rank_list,
                                              int64_t frame_count) {
    // VideoReader 在关键帧索引就绪后调用：chunk expected = rank 差。
    std::lock_guard<std::mutex> lk(mtx_);
    kf_pts_ = std::move(pts_list);
    kf_rank_ = std::move(rank_list);
    frame_count_ = frame_count;
    // pts/每帧 线性估计（CFR 下精确；VFR 为均值近似，调度启发式可接受）
    if (kf_pts_.size() >= 2 && kf_rank_.back() > kf_rank_.front()
            && kf_pts_.back() > kf_pts_.front()) {
        pts_per_frame_ = static_cast<double>(kf_pts_.back() - kf_pts_.front())
            / static_cast<double>(kf_rank_.back() - kf_rank_.front());
    }
}


void HybridThreadedDecoder::SetRoi(int x1, int y1, int x2, int y2) {
    // CPU 侧 filter 图支持热切换；GPU 侧输出池尺寸固定（与 VideoReader
    // 的 kDLCUDA 守卫同语义）—— ROI 必须在任何帧解码前固化。
    // 无效矩形（无法构造偶数超集等）：解码器回退全帧输出，池保持原状。
    cpu_.SetRoi(x1, y1, x2, y2);
    if (gpu_) gpu_->SetRoi(x1, y1, x2, y2);
    int w = x2 - x1, h = y2 - y1;
    if (w > 0 && h > 0) {
        if (frames_out_[0].load() != 0 || frames_out_[1].load() != 0
                || !emit_queue_.empty() || !gpu_pkt_q_.empty()
                || !ready_.empty() || !cpu_ready_.empty()) {
            LOG(FATAL) << "hybrid SetRoi must be called before any decode "
                       << "(frames cpu=" << frames_out_[0].load()
                       << " gpu=" << frames_out_[1].load() << ")";
        }
        // GPU 侧此后只输出 ROI 窗口：池形状/落地预算随 ROI 重建。
        // 在途异步拷贝同步后丢弃（帧尺寸已变，旧缓冲不可再收割）。
        AbortInflight();
        gpu_frame_shape_ = FrameShapeFor(output_format_, h, w);
        frame_bytes_ = 1;
        for (int64_t d : gpu_frame_shape_) frame_bytes_ *= d;
        // ROI 后输出帧仅 ROI 大小（如 1080p NV12 全帧 ~3.1MB → ROI
        // ~5.4KB）：按 ROI 帧字节重算 GPU 池/上载池深度。原深度按全帧
        // 字节预算，ROI 场景虚小 ~570× —— FeedStep 喂包闸
        // （ready_ + reserve >= gpu_pool_frames_）在 GPU 提前解码约千帧
        // 后长期关闭：CPU 块期间拉取掉到 dav1d 速率、GPU 空有已路由包
        // 不喂（[hybrid-w] idle rdy≈1006 q≈786 轨迹实测），混跑吞吐
        // 退化近交替（av1 损耗 29.5%）。ROI 帧 5.4KB × 8192 ≈ 44MB。
        {
            auto clampi = [](double v, int lo, int hi) {
                return (int)std::max<double>(double(lo),
                                             std::min<double>(v, double(hi)));
            };
            const double fb_roi = (double)frame_bytes_;
            if (out_cuda_ && fb_roi > 0) {
                gpu_pool_frames_ = clampi(vram_budget_ * 0.65 / fb_roi,
                                          96, 8192);
                up_pool_frames_ = clampi(vram_budget_ * 0.35 / fb_roi,
                                         48, 4096);
                ready_cap_frames_ = gpu_pool_frames_ + up_pool_frames_ + 64;
                // CPU 臂银行（2026-09-13 缺口分解）：ComputeBudgets 的
                // queue_frames_ 按**全帧**字节算出 1536 帧帽（4.8GB RAM 时代
                // 的保守界），但 ROI 输出下 frame 项只有 KB 级 —— 1536 帧
                // 仅 ~8MB，对侧（GPU）值日期的发射停顿盖不住（h264 GPU 头
                // 阻塞 1.39s 期间 CPU 臂银行只够 ~0.5s，实测 27% 闲置）。
                // 重算：ROI 帧按 RAM 预算放开到 4096；解码线程背压解耦到
                // raw 队列并按全帧字节预算（768MB）封顶 —— 全帧内存风险
                // 由 raw 帽独立承担，ROI 银行不再被合并计价错杀。
                queue_frames_ = clampi(ram_budget_ * 0.45 / fb_roi, 96, 4096);
                cpu_.SetQueueDepth(queue_frames_);
                const int64_t raw_bytes = static_cast<int64_t>(width_) > 0
                    && static_cast<int64_t>(height_) > 0
                        ? static_cast<int64_t>(width_) * height_ * 3 / 2
                        : 3 * 1024 * 1024;
                cpu_.SetRawQueueFrames(clampi(
                    768.0 * 1024 * 1024 / static_cast<double>(raw_bytes),
                    128, 4096));
            }
        }
        gpu_pool_.Reset(gpu_pool_frames_,
                        gpu_frame_shape_, kUInt8,
                        DLDevice{kDLCUDA, device_id_});
        if (out_cuda_) {
            up_pool_.Reset(up_pool_frames_, gpu_frame_shape_, kUInt8,
                           DLDevice{kDLCUDA, device_id_});
        } else if (pinned_pool_) {
            // ROI 重建：帧尺寸已变，池按新尺寸重建（在途旧块归还时
            // 按尺寸直接释放）
            pinned_pool_->Reset(
                static_cast<std::size_t>(ready_cap_frames_)
                    + kD2HRingSlots + 16,
                static_cast<std::size_t>(frame_bytes_), gpu_frame_shape_);
        }
    }
}


void HybridThreadedDecoder::Start() {
    if (getenv("DECORD_HYBRID_FLIGHT")) {
        FILE *lf = fopen("decord_hybrid_flight.log", "a");
        if (lf) {
            fprintf(lf, "# start lander_run=%d flight_run=%d\n",
                    (int)lander_run_.load(), (int)flight_run_.load());
            fclose(lf);
        }
    }
#if defined(_WIN32)
    std::vector<unsigned long> pre_tids;
    if (affinity_on_) pre_tids = SnapshotThreads();
#endif
    cpu_.Start();
    if (gpu_) gpu_->Start();
    if (!lander_run_.load()) {
        gpu_pool_.Start();
        up_pool_.Start();  // 与 gpu_pool_ 对称：Clear/Stop 后恢复
        lander_run_.store(true);
        static std::once_flag term_flag;
        std::call_once(term_flag, [] {
            std::set_terminate([] {
                if (auto ep = std::current_exception()) {
                    try {
                        std::rethrow_exception(ep);
                    } catch (const std::exception &e) {
                        fprintf(stderr,
                                "[decord] TERMINATE: uncaught exception: %s\n",
                                e.what());
                    } catch (...) {
                        fprintf(stderr,
                                "[decord] TERMINATE: unknown exception\n");
                    }
                } else {
                    fprintf(stderr, "[decord] TERMINATE called\n");
                }
                fflush(stderr);
                std::abort();
            });
        });
        std::thread t(&HybridThreadedDecoder::GpuWorkerLoop, this);
        lander_ = std::move(t);
        if (out_cuda_) {
            if (up_stream_ == nullptr) {
                // blocking 流（flags=0）：与 legacy 默认流（消费侧 D2D/D2H
                // 拷贝所在）隐式互斥序 —— 非阻塞流下，上载缓冲回池复用时
                // 的 H2D 会与消费侧已提交未执行的拷贝读-写竞态（实测
                // CPU chunk 边界帧偶发内容差）。与 CU 解码器 stream_
                // 同为 blocking，两流间无互斥，NVDEC 不被上载阻塞。
                cudaStreamCreateWithFlags(
                    reinterpret_cast<cudaStream_t *>(&up_stream_), 0);
            }
            std::thread u(&HybridThreadedDecoder::UploaderLoop, this);
            uploader_ = std::move(u);
        }
        // 飞行记录器（取证用，env 门控；消费者路径零接触）：每实例一个
        // 线程，4Hz 快照写 decord_hybrid_flight.log（hybrid/recorder.cc）
        if (getenv("DECORD_HYBRID_FLIGHT") && !flight_run_.load()) {
            flight_run_.store(true);
            std::thread ft(&HybridThreadedDecoder::FlightRecorderLoop, this);
            flight_th_ = std::move(ft);
        }
    }
#if defined(_WIN32)
    if (affinity_on_) {
        // 差分本轮新增线程：lander/uploader → service；其余新增（cpu_
        // 解码 worker / filter worker）→ decode。此前直接把存量+新增
        // 全归 service，把 cpu_ worker 也钉去服务核 → CPU 臂饿死。
        const unsigned long lander_tid = lander_.joinable()
            ? GetThreadId(lander_.native_handle()) : 0;
        const unsigned long uploader_tid = uploader_.joinable()
            ? GetThreadId(uploader_.native_handle()) : 0;
        auto after = SnapshotThreads();
        std::sort(after.begin(), after.end());
        std::sort(pre_tids.begin(), pre_tids.end());
        std::vector<unsigned long> added;
        std::set_difference(after.begin(), after.end(),
                            pre_tids.begin(), pre_tids.end(),
                            std::back_inserter(added));
        for (DWORD tid : added) {
            if (tid == lander_tid || tid == uploader_tid) {
                SetThreadAffinity(tid, service_mask_);
            } else {
                if (SetThreadAffinity(tid, decode_mask_)) {
                    decode_tids_.push_back(tid);
                }
            }
        }
        PinRemainingToService();   // 存量（python/TRT/引擎）
    }
#endif
}


void HybridThreadedDecoder::StopGpuWorker() {
    bool dbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
    if (getenv("DECORD_HYBRID_FLIGHT")) {
        FILE *lf = fopen("decord_hybrid_flight.log", "a");
        if (lf) { fprintf(lf, "# gpu-worker-stop\n"); fclose(lf); }
    }
    lander_run_.store(false);
    lcv_.notify_all();
    gpu_pool_.Stop();  // 唤醒阻塞在 Acquire 的工作线程
    up_pool_.Stop();
    flight_run_.store(false);   // 记录器先停（10ms 步进退出，先于 join）
    if (flight_th_.joinable()) {
        flight_th_.join();
    }
    if (dbg) fprintf(stderr, "[hybrid-d] joining worker\n");
    if (lander_.joinable()) {
        lander_.join();
    }
    if (dbg) fprintf(stderr, "[hybrid-d] joining uploader\n");
    if (uploader_.joinable()) {
        uploader_.join();
    }
    // 在途异步拷贝同步后丢弃（工作线程已 join，独占访问安全）
    AbortInflight();
    if (up_stream_ != nullptr) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(up_stream_));
        up_stream_ = nullptr;
    }
    for (auto &slot : up_staging_) {
        if (slot != nullptr) {
            cudaFreeHost(slot);
            slot = nullptr;
        }
    }
    up_stage_bytes_ = 0;
    if (d2h_stream_ != nullptr) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(d2h_stream_));
        d2h_stream_ = nullptr;
    }
    for (auto &slot : d2h_staging_) {
        if (slot != nullptr) {
            cudaFreeHost(slot);
            slot = nullptr;
        }
    }
    d2h_bytes_ = 0;
    for (auto &ev : d2h_ev_) {
        if (ev != nullptr) {
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev));
            ev = nullptr;
        }
    }
    if (dbg) fprintf(stderr, "[hybrid-d] worker joined\n");
}


void HybridThreadedDecoder::SetDecodeWindow(int64_t max_frames) {
    // Start 前由消费线程调用；与 NeedsPackets/PumpFeed 同线程，无竞态。
    window_frames_ = max_frames > 0 ? max_frames : -1;
}


bool HybridThreadedDecoder::HasPartialSupplyLocked() const {
    // 游标 GOP 已派侧且未供完（主区间未到已见末端，或还有迟到包）→
    // 边界 GOP 仍需 demux 供包（硬窗的解码语义例外）。
    if (feed_gop_idx_ >= static_cast<int64_t>(gops_.size())) return false;
    const GopRec &g = gops_[feed_gop_idx_];
    if (g.side == Side(-1)) return false;   // 未派侧 = 窗外，不供
    const int64_t avail_end = g.closed ? g.pkt_end : cache_seq_;
    return g.fed_upto < avail_end
        || (g.closed
            && g.straggler_idx
                   < static_cast<int64_t>(g.stragglers.size()));
}


std::string HybridThreadedDecoder::HybridStatsProbe() {
    // 与 Stop() 的 stderr 汇总同源，另补 HOL 直方图与份额轨迹（穿透轮新增）。
    // 协议 "k=v;k=v"：值全为数字（mode 用 out_cuda 0/1）；列表键值为逗号
    // 串（hol_hist_c/g 桶计数、trace 扁平四元组）。python 侧 hybrid_stats()
    // 负责解析。字符串拼装只在快照时刻发生一次（µs 级），不在热路径。
    char buf[64];
    std::string out;
    auto kv = [&](const char *k, long long v) {
        snprintf(buf, sizeof(buf), "%s=%lld;", k, v);
        out += buf;
    };
    auto kvf = [&](const char *k, double v) {
        snprintf(buf, sizeof(buf), "%s=%.3f;", k, v);
        out += buf;
    };
    kv("frames_c", frames_out_[0].load());
    kv("frames_g", frames_out_[1].load());
    kv("chunks_c", chunks_assigned_[0]);
    kv("chunks_g", chunks_assigned_[1]);
    kv("kicks_c", kicks_[0].load());
    kv("kicks_g", kicks_[1].load());
    kv("clones", fb_clones_.load());
    kv("out_cuda", out_cuda_ ? 1 : 0);
    kvf("rc_now", cpu_.ProductionRate());
    kvf("rg_now", gpu_rate_landed_.load(std::memory_order_relaxed));
    kv("cache_peak_mb", (long long)(cache_peak_bytes_ >> 20));
    kv("late", late_feeds_.load(std::memory_order_relaxed));
    kv("strag", strag_total_.load(std::memory_order_relaxed));
    kv("force_eof", force_eof_close_.load(std::memory_order_relaxed));
    kv("gpu_arm_stall", gpu_arm_stall_.load(std::memory_order_relaxed));
    kv("assigned_c", assigned_frames_[0]);
    kv("assigned_g", assigned_frames_[1]);
    kv("window_frames", window_frames_);
    kv("assigned_total", assigned_frames_[0] + assigned_frames_[1]);
    kv("hol_us_c", hol_us_[0].load());
    kv("hol_ev_c", hol_ev_[0].load());
    kv("strandmax_c", hol_strand_max_[0].load());
    kv("hol_us_g", hol_us_[1].load());
    kv("hol_ev_g", hol_ev_[1].load());
    kv("strandmax_g", hol_strand_max_[1].load());
    // HOL 直方图（log2 桶：episode 入口对侧存货的分布形状）
    for (int s = 0; s < 2; ++s) {
        out += s == 0 ? "hol_hist_c=" : "hol_hist_g=";
        for (int b = 0; b < 24; ++b) {
            snprintf(buf, sizeof(buf), "%s%lld", b ? "," : "",
                     hol_hist_[s][b].load(std::memory_order_relaxed));
            out += buf;
        }
        out += ";";
    }
    kv("busy_cpu_us", cpu_.DecodeBusyUs());
    kv("busy_cpu_pkts", cpu_.DecodeBusyPkts());
    kv("busy_gpu_us", gpu_ ? gpu_->DecodeBusyUs() : 0);
    kv("busy_gpu_pics", gpu_ ? gpu_->DecodeBusyPics() : 0);
    if (out_cuda_) {
        const long long fn = up_flush_n_.load();
        const long long fr = up_flush_f_.load();
        kv("up_flushes", fn);
        kv("up_frames", fr);
        kvf("up_avg_batch", fn > 0 ? (double)fr / (double)fn : 0.0);
        kv("up_nobuf", up_nobuf_.load());
        kv("up_cempty", up_cempty_.load());
    }
    // 份额轨迹（仅 DECORD_HYBRID_TRACE=1 且有样本时；扁平 csv 四元组）
    const size_t tn = trace_n_.load(std::memory_order_relaxed);
    if (trace_on_ && tn > 0) {
        out += "trace=";
        for (size_t i = 0; i < tn; ++i) {
            snprintf(buf, sizeof(buf), "%s%.0f,%.0f,%.1f,%.1f", i ? ";" : "",
                     trace_ring_[i * 4], trace_ring_[i * 4 + 1],
                     trace_ring_[i * 4 + 2], trace_ring_[i * 4 + 3]);
            out += buf;
        }
        // trace 用 ';' 分隔四元组、',' 分隔字段（与 kv 的 ';' 终结符冲突：
        // 放最后并省略终结 ';'，python 侧按 'trace=' 前缀单独解析）
    }
    return out;
}


void HybridThreadedDecoder::Stop() {
    if (getenv("DECORD_HYBRID_DEBUG")) {
        fprintf(stderr, "[hybrid] chunks cpu=%d gpu=%d frames cpu=%lld gpu=%lld\n",
                chunks_assigned_[0], chunks_assigned_[1],
                (long long)frames_out_[0].load(), (long long)frames_out_[1].load());
    }
    // 先停工作线程（它可能正持有 GPU 侧的包/缓冲），再停子解码器
    StopGpuWorker();
    if (getenv("DECORD_HYBRID_STATS")) {
        // 一次性汇总（非打印测量协议的唯一输出点；析构时触发，不在被测
        // 墙钟窗口内）。口径见 .h「非打印测量」注释。
        fprintf(stderr,
                "\n[hybrid-stats] mode=%s frames c=%lld g=%lld chunks c=%d g=%d"
                " kicks c=%lld g=%lld clones=%lld\n",
                out_cuda_ ? "gpu-out" : "cpu-out",
                (long long)frames_out_[0].load(), (long long)frames_out_[1].load(),
                chunks_assigned_[0], chunks_assigned_[1],
                (long long)kicks_[0].load(), (long long)kicks_[1].load(),
                (long long)fb_clones_.load());
        fprintf(stderr,
                "[hybrid-stats] disp gops c=%d g=%d cache_peak=%zuMB late=%lld strag=%lld rc_now=%.0f"
                " rg_now=%.0f assigned c=%lld g=%lld\n",
                chunks_assigned_[0], chunks_assigned_[1],
                cache_peak_bytes_ >> 20,
                (long long)late_feeds_.load(std::memory_order_relaxed),
                (long long)strag_total_.load(std::memory_order_relaxed),
                cpu_.ProductionRate(),
                gpu_rate_landed_.load(std::memory_order_relaxed),
                (long long)assigned_frames_[0], (long long)assigned_frames_[1]);
        fprintf(stderr,
                "[hybrid-stats] hol cpu-head us=%lld ev=%lld strandmax=%lld"
                " | gpu-head us=%lld ev=%lld strandmax=%lld\n",
                (long long)hol_us_[0].load(), (long long)hol_ev_[0].load(),
                (long long)hol_strand_max_[0].load(),
                (long long)hol_us_[1].load(), (long long)hol_ev_[1].load(),
                (long long)hol_strand_max_[1].load());
        // 忙时分解（2026-09-12 可见性）：cpu_us/gpu_us = 两臂各自实际
        // 解码墙钟（CPU=worker 取包→解码完成；GPU=cuvidDecodePicture
        // 提交时长）。与 frames c/g 联合得每臂有效解码速率（busy fps）；
        // 与墙钟联合得每臂忙碌占比 —— 「臂本身慢」（busy fps 低）与
        // 「臂被调度闲置」（busy 占比低）自此可分。
        fprintf(stderr,
                "[hybrid-stats] busy cpu_us=%lld pkts=%lld"
                " | gpu_us=%lld pics=%lld\n",
                (long long)cpu_.DecodeBusyUs(),
                (long long)cpu_.DecodeBusyPkts(),
                (long long)(gpu_ ? gpu_->DecodeBusyUs() : 0),
                (long long)(gpu_ ? gpu_->DecodeBusyPics() : 0));
        if (out_cuda_) {
            const long long fn = (long long)up_flush_n_.load();
            const long long fr = (long long)up_flush_f_.load();
            fprintf(stderr,
                    "[hybrid-stats] upload flushes=%lld frames=%lld avg_batch=%.2f"
                    " nobuf=%lld cpuempty=%lld\n",
                    fn, fr, fn > 0 ? (double)fr / (double)fn : 0.0,
                    (long long)up_nobuf_.load(), (long long)up_cempty_.load());
        }
    }
    cpu_.Stop();
    if (gpu_) gpu_->Stop();
}


void HybridThreadedDecoder::Clear() {
    StopGpuWorker();
    cpu_.Clear();
    if (gpu_) gpu_->Clear();
    ResetRouting();
}


void HybridThreadedDecoder::ResetRouting() {
    std::lock_guard<std::mutex> lk(mtx_);
    routing_active_ = false;
    kick_side_ = Side(-1);
    kick_owed_ = kick_cloned_ = 0;
    emit_queue_.clear();
    stash_[0] = runtime::NDArray();
    stash_[1] = runtime::NDArray();
    has_stash_[0] = has_stash_[1] = false;
    eof_pushed_ = false;
    chunks_assigned_[0] = chunks_assigned_[1] = 0;
    assigned_frames_[0] = assigned_frames_[1] = 0;
    pkt_cache_.clear();
    cache_seq_ = 0;
    cache_base_seq_ = 0;
    cache_bytes_ = 0;
    gops_.clear();
    feed_gop_idx_ = 0;
    gop_seq_ = 0;
    last_fed_side_ = Side(-1);
    eof_cache_ = false;
    arm_flush_sent_ = false;
    eof_flush_out_ = false;
    emitted_total_ = 0;
    sched_initialized_ = false;
    side_pending_[0] = side_pending_[1] = 0;
    {
        std::lock_guard<std::mutex> lk2(lcv_mtx_);
        gpu_pkt_q_.clear();
        gpu_flush_left_ = 0;
    }
    {
        std::lock_guard<std::mutex> lk2(rmtx_);
        ready_.clear();
        cpu_ready_.clear();
    }
    // kf 索引保留：Seek 后复用帧数表
}


#if defined(_WIN32)
#include <tlhelp32.h>
namespace {
std::vector<unsigned long> SnapshotThreads() {
    std::vector<unsigned long> out;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return out;
    THREADENTRY32 te{}; te.dwSize = sizeof(te);
    DWORD pid = GetCurrentProcessId();
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid && te.th32ThreadID != 0) {
                out.push_back(te.th32ThreadID);
            }
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    return out;
}
bool SetThreadAffinity(DWORD tid, uintptr_t mask) {
    HANDLE h = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION,
                          FALSE, tid);
    if (!h) return false;
    BOOL ok = SetThreadAffinityMask(h, static_cast<DWORD_PTR>(mask));
    CloseHandle(h);
    return ok != FALSE;
}
}  // namespace
#endif  // _WIN32

void HybridThreadedDecoder::InitAffinityMasks() {
#if defined(_WIN32)
    // env：DECORD_HYBRID_DECODE_CORES=物理核数（0/未设=关；-1=默认
    // total-3）。仅在单 processor group（逻辑核 ≤64）平台生效。
    int want = 0;
    if (const char *e = getenv("DECORD_HYBRID_DECODE_CORES")) want = atoi(e);
    if (want == 0) return;   // 未设/0 = 关（引擎在 TRT 路径显式开）
    if (want < 0) want = -1;  // -1 = 默认 total-3 分区
    DWORD need = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &need)
            && GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return;
    }
    std::vector<char> buf(need);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
            &need)) {
        return;
    }
    std::vector<uintptr_t> core_masks;   // 每物理核的 SMT 掩码
    size_t off = 0;
    while (off < need) {
        auto *pi = reinterpret_cast<
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (pi->Relationship == RelationProcessorCore) {
            uintptr_t m = 0;
            for (WORD g = 0; g < pi->Processor.GroupCount; ++g) {
                if (pi->Processor.GroupMask[g].Group == 0) {
                    m |= static_cast<uintptr_t>(
                        pi->Processor.GroupMask[g].Mask);
                }
            }
            if (m) core_masks.push_back(m);
        }
        off += pi->Size;
    }
    if (core_masks.empty()) return;
    const int total = static_cast<int>(core_masks.size());
    int ndec = want > 0 ? want : (total - 3);
    if (ndec < 2) ndec = 2;
    if (ndec > total - 1) ndec = total - 1;   // service 至少 1 物理核
    uintptr_t dm = 0, sm = 0;
    for (int i = 0; i < total; ++i) {
        if (i < ndec) dm |= core_masks[i]; else sm |= core_masks[i];
    }
    if (!dm || !sm) return;
    decode_mask_ = dm;
    service_mask_ = sm;
    affinity_on_ = true;
    if (getenv("DECORD_HYBRID_DEBUG")) {
        fprintf(stderr, "[hybrid-affinity] phys=%d decode=%llu service=%llu\n",
                total, (unsigned long long)dm, (unsigned long long)sm);
    }
#endif
}


bool HybridThreadedDecoder::Drained() const {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!emit_queue_.empty() || has_stash_[0] || has_stash_[1]) return false;
    }
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        if (!ready_.empty()) return false;
        if (out_cuda_ && !cpu_ready_.empty()) return false;
    }
    {
        std::lock_guard<std::mutex> lk(lcv_mtx_);
        if (!gpu_pkt_q_.empty() || gpu_flush_left_ > 0) return false;
    }
    if (!cpu_.Drained()) return false;
    if (gpu_ && !gpu_->Drained()) return false;
    return true;
}


void HybridThreadedDecoder::SuggestDiscardPTS(std::vector<int64_t> dts) {
    cpu_.SuggestDiscardPTS(dts);
    if (gpu_) gpu_->SuggestDiscardPTS(dts);
}


void HybridThreadedDecoder::ClearDiscardPTS() {
    cpu_.ClearDiscardPTS();
    if (gpu_) gpu_->ClearDiscardPTS();
}

void HybridThreadedDecoder::PinFfmpegThreads() {
#if defined(_WIN32)
    if (!affinity_on_) return;
    auto after = SnapshotThreads();
    std::sort(after.begin(), after.end());
    std::vector<unsigned long> added;
    std::set_difference(after.begin(), after.end(),
                        aff_baseline_.begin(), aff_baseline_.end(),
                        std::back_inserter(added));
    int pinned = 0;
    for (DWORD tid : added) {
        if (SetThreadAffinity(tid, decode_mask_)) {
            decode_tids_.push_back(tid);
            ++pinned;
        }
    }
    if (getenv("DECORD_HYBRID_DEBUG")) {
        fprintf(stderr, "[hybrid-affinity] pinned %d ffmpeg threads to decode set\n",
                pinned);
    }
#endif
}

void HybridThreadedDecoder::PinRemainingToService() {
#if defined(_WIN32)
    if (!affinity_on_) return;
    std::sort(decode_tids_.begin(), decode_tids_.end());
    auto all = SnapshotThreads();
    std::sort(all.begin(), all.end());
    std::vector<unsigned long> others;
    std::set_difference(all.begin(), all.end(),
                        decode_tids_.begin(), decode_tids_.end(),
                        std::back_inserter(others));
    int pinned = 0;
    for (DWORD tid : others) {
        if (SetThreadAffinity(tid, service_mask_)) ++pinned;
    }
    if (getenv("DECORD_HYBRID_DEBUG")) {
        fprintf(stderr, "[hybrid-affinity] %d threads -> service set\n", pinned);
    }
#endif
}


}  // namespace decord
