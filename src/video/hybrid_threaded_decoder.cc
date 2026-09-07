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
 *  区间（实测双发错位），故 AV1 固定单侧 GPU（见 ChooseSide）。
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

#include "hybrid_threaded_decoder.h"

#include "ffmpeg/ffmpeg_common.h"
#include <decord/runtime/ndarray.h>

#ifdef DECORD_USE_CUDA
#include "nvcodec/cuda_threaded_decoder.h"
#endif

#include <dmlc/logging.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

namespace decord {

namespace {
/*! 调度粘性：同侧最少连续分配的 chunk 数（块状分配，防流水线冷启动） */
constexpr int kStickyMinChunks = 4;
}  // namespace

HybridThreadedDecoder::HybridThreadedDecoder(int device_id,
                                             AVCodecParameters *codecpar,
                                             const AVInputFormat *iformat,
                                             bool output_cuda)
    : cpu_(), device_id_(device_id), out_cuda_(output_cuda) {
#ifdef DECORD_USE_CUDA
    // GPU 子解码器初始化 bsf (mp4→annexb) 时会就地改写传入的 codecpar。
    // 传副本；CPU 侧继续用 VideoReader 手里的原始 AVCC 参数（其
    // avcodec_open2 发生在本构造之后），GPU 侧后续用被转换的副本。
    gpu_codecpar_.reset(avcodec_parameters_alloc());
    CHECK(gpu_codecpar_ != nullptr) << "avcodec_parameters_alloc failed";
    CHECK_GE(avcodec_parameters_copy(gpu_codecpar_.get(), codecpar), 0)
        << "avcodec_parameters_copy failed";
    gpu_.reset(new cuda::CUThreadedDecoder(device_id, gpu_codecpar_.get(), iformat));
#else
    LOG(FATAL) << "HybridThreadedDecoder requires DECORD_USE_CUDA build";
#endif
}

HybridThreadedDecoder::~HybridThreadedDecoder() {
    if (getenv("DECORD_HYBRID_DEBUG")) fprintf(stderr, "[hybrid-d] enter\n");
    Stop();
    if (getenv("DECORD_HYBRID_DEBUG")) fprintf(stderr, "[hybrid-d] stopped\n");
}

#ifdef DECORD_USE_CUDA
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

void HybridGpuBufferPool::EnableReleaseSync() {
    if (release_ev_ != nullptr) return;
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess) {
        release_ev_ = ev;
    }
}

void HybridGpuBufferPool::WaitRelease(void *stream) {
    if (release_ev_ != nullptr) {
        sync_stream_ = stream;
        cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream),
                            reinterpret_cast<cudaEvent_t>(release_ev_), 0);
    }
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
#endif  // DECORD_USE_CUDA

void HybridThreadedDecoder::ComputeBudgets() {
    // adaptive: free VRAM/RAM -> bounded budgets; alloc failure = backpressure
    double vram_budget = 768.0 * 1024 * 1024;
    double ram_budget = 1536.0 * 1024 * 1024;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b > 0)
        vram_budget = static_cast<double>(free_b) * 0.30;
#if defined(_WIN32)
    { MEMORYSTATUSEX ms{}; ms.dwLength = sizeof(ms);
      if (GlobalMemoryStatusEx(&ms)) ram_budget = static_cast<double>(ms.ullAvailPhys) * 0.30; }
#else
    { long pages = sysconf(_SC_AVPHYS_PAGES); long ps = sysconf(_SC_PAGE_SIZE);
      if (pages > 0 && ps > 0) ram_budget = static_cast<double>(pages) * ps * 0.30; }
#endif
    if (const char *e = getenv("DECORD_HYBRID_VRAM_BUDGET_MB"))
        if (atof(e) > 0) vram_budget = atof(e) * 1024 * 1024;
    if (const char *e = getenv("DECORD_HYBRID_RAM_BUDGET_MB"))
        if (atof(e) > 0) ram_budget = atof(e) * 1024 * 1024;
    const double fb = frame_bytes_ > 0 ? static_cast<double>(frame_bytes_) : 3.1e6;
    auto clampi = [](double v, int lo, int hi) {
        return static_cast<int>(std::max<double>(lo, std::min<double>(v, hi))); };
    if (out_cuda_) {
        gpu_pool_frames_ = clampi(vram_budget * 0.65 / fb, 96, 1024);
        up_pool_frames_ = clampi(vram_budget * 0.35 / fb, 48, 512);
    } else {
        gpu_pool_frames_ = clampi(vram_budget * 0.65 / fb, 28, 128);
        up_pool_frames_ = 0;
    }
    queue_frames_ = clampi(ram_budget * 0.45 / fb, 96, 768);
    ready_cap_frames_ = clampi(ram_budget * 0.45 / fb, 96, 2048);
    if (out_cuda_) ready_cap_frames_ = gpu_pool_frames_ + up_pool_frames_ + 64;
    // demux 领先必须同时覆盖两路存货（CPU queue + GPU ready）+ 余量：
    // 否则两路分食领先窗口互相饿（CPU 块发射期 GPU 拿不到包只能空转，
    // av1 6000 帧实测退化到 836fps）。压缩包驻留 RAM 仅 ~17KB/帧。
    prefetch_frames_ = clampi(queue_frames_ + ready_cap_frames_ + 128,
                              192, 3072);
    if (codec_id_ == AV_CODEC_ID_AV1) {
        // AV1：跨侧交界（CPU 块尾 → GPU 块首）存在既有内容竞态，GPU
        // 在交界前数百 ms 已深超前解码会放大它（交界前 GPU 领先
        // ~2000 帧 vs 基线 ~84 帧，实测交界尾 8 帧错位偶发）。AV1
        // 全 GPU 后 384 领先对吞吐无约束（发射吃 NVDEC 实时速率）。
        prefetch_frames_ = std::min(prefetch_frames_, 384);
    }

    if (getenv("DECORD_HYBRID_DEBUG")) {
        fprintf(stderr, "[hybrid-budget] vram=%.0fMB ram=%.0fMB pool=%d up=%d queue=%d ready=%d prefetch=%d\n",
                vram_budget / 1048576.0, ram_budget / 1048576.0,
                gpu_pool_frames_, up_pool_frames_, queue_frames_,
                ready_cap_frames_, prefetch_frames_); }
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
#ifdef DECORD_USE_CUDA
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
#endif
    // CPU 子解码器接管 VideoReader 打开的 ctx（内部 dec_ctx_.reset 持有）
    // 深存货队列：CPU chunk 的发射靠 cpu_ready_/cpu_ 内部存货瞬时完成，
    // 默认 32 帧背压会让每个 CPU chunk 退化为实时跟随解码（hevc 0.70x）。
    cpu_.SetQueueDepth(queue_frames_);  // 存货深度与 prefetch 匹配（~1.2GB RAM@1080p）
    cpu_.SetCodecContext(dec_ctx, width, height, rotation, output_format);
#ifdef DECORD_USE_CUDA
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
        gpu_->SetOnOutput([this] { lcv_.notify_all(); });
        cpu_.SetOnOutput([this] { lcv_.notify_all(); });
        gpu_pool_.SetOnRelease([this] { lcv_.notify_all(); });
        up_pool_.SetOnRelease([this] { lcv_.notify_all(); });
        if (out_cuda_) {
            // 上载池容量 = ready_ 预算的一半（帧数）：CPU chunk 的
            // 显存帧容器，独立于 GPU 解码池（防饿死）
            up_pool_.Reset(up_pool_frames_, gpu_frame_shape_, kUInt8,
                           DLDevice{kDLCUDA, device_id_});
        }
    }
#endif
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
    // 字节预算 → 帧数上限（随分辨率自适应）。下限必须 > kGpuPoolBuffers：
    // FeedStep 的喂包闸门是 ready_ 余量 ≥ 池大小，下限过小会让 GPU 永远
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

void HybridThreadedDecoder::SetRoi(int x1, int y1, int x2, int y2) {
    // CPU 侧 filter 图支持热切换；GPU 侧输出池尺寸固定（与 VideoReader
    // 的 kDLCUDA 守卫同语义）—— ROI 必须在任何帧解码前固化。
    // 无效矩形（无法构造偶数超集等）：解码器回退全帧输出，池保持原状。
    cpu_.SetRoi(x1, y1, x2, y2);
#ifdef DECORD_USE_CUDA
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
        gpu_pool_.Reset(gpu_pool_frames_,
                        gpu_frame_shape_, kUInt8,
                        DLDevice{kDLCUDA, device_id_});
        if (out_cuda_) {
            up_pool_.Reset(up_pool_frames_, gpu_frame_shape_, kUInt8,
                           DLDevice{kDLCUDA, device_id_});
        }
    }
#endif
}

void HybridThreadedDecoder::Start() {
    started_.store(true);
    cpu_.Start();
#ifdef DECORD_USE_CUDA
    if (gpu_) gpu_->Start();
    if (!lander_run_.load()) {
        gpu_pool_.Start();
        up_pool_.Start();  // 与 gpu_pool_ 对称：Clear/Stop 后恢复
        lander_run_.store(true);
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
    }
#endif
}

#ifdef DECORD_USE_CUDA
void HybridThreadedDecoder::StopGpuWorker() {
    bool dbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
    lander_run_.store(false);
    lcv_.notify_all();
    gpu_pool_.Stop();  // 唤醒阻塞在 Acquire 的工作线程
    up_pool_.Stop();
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
    up_stage_idx_ = 0;
    for (auto &ev : up_ev_) {
        if (ev != nullptr) {
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev));
            ev = nullptr;
        }
    }
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
#endif

void HybridThreadedDecoder::Stop() {
    started_.store(false);
    if (getenv("DECORD_HYBRID_DEBUG")) {
        fprintf(stderr, "[hybrid] chunks cpu=%d gpu=%d frames cpu=%lld gpu=%lld\n",
                chunks_assigned_[0], chunks_assigned_[1],
                (long long)frames_out_[0].load(), (long long)frames_out_[1].load());
    }
#ifdef DECORD_USE_CUDA
    // 先停工作线程（它可能正持有 GPU 侧的包/缓冲），再停子解码器
    StopGpuWorker();
#endif
    cpu_.Stop();
#ifdef DECORD_USE_CUDA
    if (gpu_) gpu_->Stop();
#endif
}

void HybridThreadedDecoder::Clear() {
#ifdef DECORD_USE_CUDA
    StopGpuWorker();
#endif
    cpu_.Clear();
#ifdef DECORD_USE_CUDA
    if (gpu_) gpu_->Clear();
#endif
    ResetRouting();
}

void HybridThreadedDecoder::ResetRouting() {
    std::lock_guard<std::mutex> lk(mtx_);
    routing_active_ = false;
    cur_side_ = SIDE_CPU;
    cur_start_pts_ = 0;
    emit_queue_.clear();
    stash_[0] = runtime::NDArray();
    stash_[1] = runtime::NDArray();
    has_stash_[0] = has_stash_[1] = false;
    eof_pushed_ = false;
    chunks_assigned_[0] = chunks_assigned_[1] = 0;
    emitted_total_ = 0;
    sticky_frames_ = 0;
    sched_initialized_ = false;
    side_pending_[0] = side_pending_[1] = 0;
#ifdef DECORD_USE_CUDA
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
    gpu_pending_ = 0;
#endif
    // kf 索引保留：Seek 后复用帧数表
}

HybridThreadedDecoder::Side HybridThreadedDecoder::ChooseSide(int64_t key_pts) {
    {/* 诊断/实验开关：DECORD_HYBRID_FORCE_SIDE=cpu|gpu 强制单侧路由 */
        static const Side forced = [] {
            const char *e = getenv("DECORD_HYBRID_FORCE_SIDE");
            if (!e) return Side(-1);
            return (strcmp(e, "gpu") == 0) ? SIDE_GPU : SIDE_CPU;
        }();
        if (forced == SIDE_GPU) return SIDE_GPU;
        if (forced == SIDE_CPU) return IsIdrLikeCodec() ? SIDE_CPU : SIDE_GPU;
    }
    // 设计决策（用户拍板，2026-09-07）：所有 codec 统一按实测速率比例
    // 分配（water-filling），**不做慢侧自动回退** —— hybrid 是实验性
    // 接口，用户已被告知可能更慢；内部尊重用户选择（decord.hybrid 显式
    // 选它 = 明确要求 CPU+GPU 混跑），即使 hevc/av1 混跑实测劣于全 GPU
    // （rc 是隔离产率，混跑时软解与消费/喂包线程抢核致有效供给缩水 +
    // 块交替发射固有损耗；自动回退会让"选了 hybrid"静默变成"跑 gpu"，
    // 实验语义失真）。后续优化方向是压低混跑损耗而非代用户跳过。
    // （历史：AV1 曾因 dav1d CPU 块尾滞留 + kGpuAheadBytes 门锁死而走
    // 零 CPU 块回避；两者均已移除/修复，AV1 恢复标准混跑路径。）
    // （GPU 待发射字节硬预算已移除）总在途由 demux prefetch 窗口 +
    // ready_ 预算 + 池深三重硬上限兜底。旧的 kGpuAheadBytes 预算把
    // "CPU 块实时发射期 GPU 存货正常偏斜"误判为调度失衡并强制切给
    // 慢侧 CPU —— GPU 越快越早触发，恶性循环锁死在 CPU 实时速率
    // （av1 全量实测 654fps vs 全 GPU 1520）。
    // 积压排空贪心 → 已被 water-filling 份额计数器取代（见下）。
    // CPU 速率取生产侧实测（filter 线程产出 EWMA，NV12 直出后的有效
    // 供给口径）：与 GPU 落地速率同口径可比。
    double rate[2] = {cpu_.ProductionRate(),
                      gpu_rate_landed_.load(std::memory_order_relaxed)};
    // rc 未学得：给一个 CPU chunk 采样（所有 codec —— water-filling 需要
    // rc）。该 chunk 与 chunk0 的 GPU 发射重叠，且 GPU-first 下盲决策全落
    // GPU，采样安全（旧版曾因水位门控屏蔽速率学习把整条流钉死 CPU）。
    if (rate[SIDE_CPU] <= 0) {
        if (getenv("DECORD_HYBRID_DEBUG")) {
            fprintf(stderr, "[hybrid-sched] key=%lld rc-unknown -> CPU-sample\n",
                    (long long)key_pts);
        }
        return SIDE_CPU;
    }
    if (rate[SIDE_GPU] <= 0) {
        // rg 未学得（决策跑在解码前面，NVDEC 落地慢一拍）：交替试探，
        // 让两侧份额从对称起点再平衡（盲阶段单边灌满 alloc 会让
        // water-filling 失去再平衡能力）。
        Side pick = chunks_assigned_[SIDE_CPU] <= chunks_assigned_[SIDE_GPU]
                        ? SIDE_CPU : SIDE_GPU;
        if (getenv("DECORD_HYBRID_DEBUG")) {
            fprintf(stderr, "[hybrid-sched] key=%lld rg-unknown rc=%.0f -> %d\n",
                    (long long)key_pts, rate[SIDE_CPU], (int)pick);
        }
        return pick;
    }
    if (!sched_initialized_) {
        // 首次双侧速率就绪：重置份额计数（盲决策阶段的单边分配不计入
        // alloc 基数，否则 GPU 的巨额"存款"让后期混跑失去再平衡能力）。
        sched_initialized_ = true;
        alloc_frames_[0] = est_chunk_frames_;
        alloc_frames_[1] = est_chunk_frames_;
    }
    // 两侧并行生产、发射吃存货，稳态吞吐 ≈ r_cpu + r_gpu（块交替的
    // 发射损耗见粘性约束；不做慢侧自动回退，见函数头设计决策）。
    // （慢侧弃用闸已按用户决策移除：曾实测 hevc/av1 混跑劣于全 GPU 而
    // 自动改道 —— 但这让 hybrid 静默退化为 gpu，实验语义失真。）
    // 份额均衡（water-filling 计数器）：目标份额 ∝ 实测速率，按"实际分配
    // 帧数与理想份额的差"决策。deficit 精确、无 tie 陷阱 —— min-max 在
    // 积压钳制成对称后两侧 work 恒等、t 恒 tie，tie->CPU 使 f_c 漂移到 1
    //（test6 全量实测 6000 帧后断崖跌到 529fps、CPU 独占、GPU 闲置）。
    // 积压安全由 GPU 字节预算 + 池耗尽背压独立兜底。
    const double est = static_cast<double>(est_chunk_frames_);
    if (getenv("DECORD_HYBRID_DEBUG")) {
        fprintf(stderr, "[hybrid-sched] key=%lld rc=%.0f rg=%.0f alloc=%lld/%lld sticky=%lld\n",
                (long long)key_pts, rate[SIDE_CPU], rate[SIDE_GPU],
                (long long)alloc_frames_[0], (long long)alloc_frames_[1],
                (long long)sticky_frames_);
    }
    const double tot = rate[SIDE_CPU] + rate[SIDE_GPU];
    const double want_cpu = (static_cast<double>(alloc_frames_[0]
                            + alloc_frames_[1]) + est) * rate[SIDE_CPU] / tot;
    Side chosen = (static_cast<double>(alloc_frames_[SIDE_CPU]) <= want_cpu)
                      ? SIDE_CPU : SIDE_GPU;
    // 粘性：同侧连续分配不足 sticky_budget 个 chunk 估计帧数则不切。
    // 下界 2 chunks：逐 chunk 交替让 CPU 侧频繁断流（dav1d 帧并行
    // 反复冷启动，有效速率掉到满速 ~1/4）；上界受 CPU 存货深度约束：
    // 块超过存货（queue_frames_）时块尾必然实时跟随解码（av1 实测
    // 4×300 帧块 > 768 存货，整体被 CPU 块实时节奏拖到 844fps，反而
    // 比全 GPU 的 1520 慢 45%）——块 ≤ 存货才能整块瞬发、另一侧的
    // 生产期覆盖本侧发射窗口。
    const int64_t cpu_budget = std::max<int64_t>(
        2 * est, std::min<int64_t>(4 * est, queue_frames_));
    // GPU 块预算按速率比放大（两侧块的生产时间对齐）：CPU 块发射期
    // GPU 生产、GPU 块发射期 CPU 攒存货 —— GPU 块过短会把 CPU 的攒货
    // 窗口压扁，CPU 块尾段退化为实时跟随（对称 768/768 块实测 av1
    // 1424fps < 全 GPU 1520）。上限 12 chunks 防极端比率。
    const int64_t gpu_budget = std::min<int64_t>(
        12 * est,
        std::max<int64_t>(cpu_budget, static_cast<int64_t>(
            cpu_budget * rate[SIDE_GPU] / std::max(rate[SIDE_CPU], 1.0))));
    const int64_t sticky_budget =
        cur_side_ == SIDE_CPU ? cpu_budget : gpu_budget;
    if (chosen != cur_side_
            && sticky_frames_ < sticky_budget
            && est > 0) {
        chosen = cur_side_;
    }
    if (chosen == cur_side_) {
        sticky_frames_ += static_cast<int64_t>(est);
    } else {
        sticky_frames_ = static_cast<int64_t>(est);
    }
    return chosen;
}

int64_t HybridThreadedDecoder::RankOfPts(int64_t pts) const {
    // pts → 呈现序帧号（kf 表 lower_bound：精确命中给表值，介于两关键
    // 帧之间给较低 rank —— 调度启发式足够）
    if (kf_pts_.empty()) return 0;
    auto it = std::lower_bound(kf_pts_.begin(), kf_pts_.end(), pts);
    if (it == kf_pts_.end()) return frame_count_;
    int64_t r = kf_rank_[it - kf_pts_.begin()];
    if (*it != pts && r > 0) return r - 1;
    return r;
}

void HybridThreadedDecoder::Push(ffmpeg::AVPacketPtr pkt, runtime::NDArray buf) {
    // VideoReader 对 hybrid 一律推 NDArray()（与 CPU 分支一致）；
    // GPU 侧需要的 CUDA 输出缓冲由工作线程从有界池 Acquire。
    if (!pkt) {
        // EOF flush
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (routing_active_) {
                Chunk last{cur_side_, cur_start_pts_, INT64_MAX, 0, 0};
                emit_queue_.push_back(last);
            }
            eof_pushed_ = true;
        }
        cpu_.Push(nullptr, runtime::NDArray());
#ifdef DECORD_USE_CUDA
        if (gpu_) {
            // GPU 侧 flush 由工作线程在包队列排空后执行（Push 永不阻塞）
            {
                std::lock_guard<std::mutex> lk(lcv_mtx_);
                gpu_flush_left_ = ThreadedDecoderInterface::kMaxOutputSurfaces;
            }
            lcv_.notify_all();
        }
#endif
        return;
    }
    bool is_key = (pkt->flags & AV_PKT_FLAG_KEY) != 0;
    Side flush_side = SIDE_CPU;
    bool need_flush = false;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!routing_active_) {
            // 首包一律 GPU：NVDEC 平价起步，chunk0 的 CPU 串行解码
            // （~0.3s @ 软解速率，hevc 全量 0.3/1.7s ≈ 15% 纯损耗）完全
            // 消除。rc 采样仅 h264 有意义（混跑有净收益）——由 rc-unknown
            // 分支按 codec 给一个 CPU chunk 采样（与 chunk0 的 GPU 发射
            // 重叠）；hevc/av1 的 gate 判决恒为全 GPU，永远不需要 rc。
            routing_active_ = true;
            cur_side_ = SIDE_GPU;
            cur_start_pts_ = pkt->pts;
            Chunk c{cur_side_, pkt->pts, INT64_MAX, 0, 0};
            emit_queue_.push_back(c);
            ++chunks_assigned_[cur_side_];
        } else if (is_key) {
            // 关闭当前 chunk（end = 本关键帧 pts），为新 chunk 选侧
            if (!emit_queue_.empty()) {
                Chunk &last = emit_queue_.back();
                last.end_pts = pkt->pts;
                last.expected = ExpectedFrames(last.start_pts, pkt->pts);
                if (last.expected > 0) {
                    est_chunk_frames_ = last.expected;
                }
            }
            Side old_side = cur_side_;
            Side s = ChooseSide(pkt->pts);
            // kick 冲刷：跨侧切换时克隆新 chunk 的关键帧包推给离场侧，
            // 驱动其交出滞留的尾部帧（NVDEC 的 display 延迟需要后续包
            // 驱动；离场侧断流后末帧滞留 → 发射端死等 → 消费侧重试把
            // 文件 demux 完、EOF 语义污染中途读取，av1 实测 599/600 帧
            // + 重复帧）。IDR 型：kick 帧重置参考状态、输出 pts == 包
            // pts（stash 闭环）。AV1：kick 帧输出可能回落上一 chunk
            // （show_existing 映射）→ 陈旧丢弃兜底；迟到的块尾帧 pts
            // 属于本 chunk → 正常发射补满 expected。
            if (s != old_side && IsIdrLikeCodec()) {
                flush_side = old_side;
                need_flush = true;
            }
            cur_side_ = s;
            cur_start_pts_ = pkt->pts;
            Chunk c{s, pkt->pts, INT64_MAX, 0, 0};
            emit_queue_.push_back(c);
            ++chunks_assigned_[s];
            alloc_frames_[s] += est_chunk_frames_;

        } else {
            // 非 key 包：按 pts 归属路由。解码序与 pts 序不一致的码流
            //（重排帧的包迟到）仍属于更早的 chunk，必须喂给那侧。
            cur_route_override_ = SIDE_CPU;
            for (auto it = emit_queue_.rbegin(); it != emit_queue_.rend(); ++it) {
                if (pkt->pts >= it->start_pts) {
                    cur_route_override_ = it->side;
                    break;
                }
            }
            if (!IsIdrLikeCodec()) {
                // 非 IDR 型（AV1）无法归属的包（show_existing 的回退 pts、
                // 无 pts 等）：跟随当前 chunk 的侧。fallback CPU 会在全 GPU
                // 码流上把孤立中段包喂给软解（libdav1d "Error parsing OBU
                // data" → 进程崩溃）；其输出旧帧由 Pop 的陈旧丢弃兜底。
                // IDR 流保持 CPU fallback：迟到重排帧喂软解状态连续。
                cur_route_override_ = cur_side_;
            }
        }
        // 各侧待发射帧计数（包粒度，1 包 = 1 帧；含在途解码与存货）：
        // kick 包与本包。gpu_pending_ 保留作 GPU 字节预算依据。
        if (need_flush && flush_side == SIDE_GPU) { ++gpu_pending_; ++side_pending_[SIDE_GPU]; }
        else if (need_flush && flush_side == SIDE_CPU) { ++side_pending_[SIDE_CPU]; }
        Side tgt = (is_key ? cur_side_ : cur_route_override_);
        ++side_pending_[tgt];
        if (tgt == SIDE_GPU) ++gpu_pending_;
    }
    if (need_flush) {
        // kick（锁外执行）。与原型"块尾多解一帧"同型：克隆新 chunk 的
        // 关键帧包推给离场侧，触发其交出滞留的重排帧。kick 帧与迟到帧
        // 的顺序由 Pop 的 expected 帧数对齐保证（kick 帧 stash 扣住，
        // 迟到帧先发）。
        AVPacket *kick = av_packet_clone(pkt.get());
        CHECK(kick != nullptr) << "av_packet_clone failed";
        ffmpeg::AVPacketPtr kick_ptr(kick, [](AVPacket *p) { av_packet_free(&p); });
        if (flush_side == SIDE_CPU) {
            cpu_.Push(std::move(kick_ptr), runtime::NDArray());
        } else {
#ifdef DECORD_USE_CUDA
            {
                std::lock_guard<std::mutex> lk(lcv_mtx_);
                gpu_pkt_q_.push_back(std::move(kick_ptr));
            }
#endif
        }
    }
    Side target = is_key ? cur_side_ : cur_route_override_;
    if (target == SIDE_CPU) {
        cpu_.Push(std::move(pkt), runtime::NDArray());
    } else {
#ifdef DECORD_USE_CUDA
        // GPU 包进宿主队列（压缩包很小），由工作线程按 ready_ 余量喂给
        // GPU —— demux 线程永不阻塞，显存占用由有界池决定。
        {
            std::lock_guard<std::mutex> lk(lcv_mtx_);
            gpu_pkt_q_.push_back(std::move(pkt));
        }
        lcv_.notify_all();
#endif
    }
}

bool HybridThreadedDecoder::IsMarker(const runtime::NDArray &f) {
    if (!f.defined()) return false;
    const DLTensor *t = f.operator->();
    return t != nullptr && t->dtype.code == kDLInt && t->dtype.bits == 64;
}

bool HybridThreadedDecoder::PopSide(Side s, runtime::NDArray *f) {
    // 取帧优先级：stash（越界暂存）→ 子解码器 / ready_ 队列
    if (has_stash_[s]) {
        *f = stash_[s];
        has_stash_[s] = false;
        return true;
    }
    if (s == SIDE_CPU) {
#ifdef DECORD_USE_CUDA
        if (out_cuda_) {
            // GPU 驻留模式：取已上载的显存帧（marker 由 UploadStep 直通），非阻塞
            std::lock_guard<std::mutex> lk(rmtx_);
            if (cpu_ready_.empty()) return false;
            *f = std::move(cpu_ready_.front());
            cpu_ready_.pop_front();
            lcv_.notify_all();  // 槽位/预算释放：即时唤醒喂包/上载线程
            return f->defined();
        }
#endif
        return cpu_.Pop(f);
    }
#ifdef DECORD_USE_CUDA
    // GPU 帧：取工作线程已落地/直通的帧，非阻塞
    std::lock_guard<std::mutex> lk(rmtx_);
    if (ready_.empty()) return false;
    *f = std::move(ready_.front());
    ready_.pop_front();
    lcv_.notify_all();  // 槽位/预算释放：即时唤醒喂包线程
    return f->defined();
#else
    return false;
#endif
}

bool HybridThreadedDecoder::Pop(runtime::NDArray *frame) {
    while (true) {
        Chunk ch;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (emit_queue_.empty()) {
                break;
            }
            ch = emit_queue_.front();
        }
        // force-close 安全网：该侧已路由的包全部出清（pending==0）但
        // expected 未补满 —— 重排流的跨侧切换残留（无 kick 的 AV1 混合
        // 路由）。接受缺失帧关闭 chunk（后续陈旧丢弃保序），避免无限等待。
        if (!eof_pushed_ && ch.expected > 0 && ch.emitted < ch.expected
                && side_pending_[ch.side] == 0) {
            bool close_it = false;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (emit_queue_.size() > 1
                        && emit_queue_.front().start_pts == ch.start_pts
                        && emit_queue_.front().emitted == ch.emitted
                        && side_pending_[ch.side] == 0) {
                    emit_queue_.pop_front();
                    close_it = true;
                }
            }
            if (close_it) {
                if (getenv("DECORD_HYBRID_DEBUG")) {
                    fprintf(stderr, "[hybrid] force-close chunk start=%lld emitted=%lld/%lld\n",
                            (long long)ch.start_pts, (long long)ch.emitted,
                            (long long)ch.expected);
                }
                continue;
            }
        }
        Side s = ch.side;
        runtime::NDArray f;
        if (!PopSide(s, &f)) {
            static const bool dbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
            if (dbg) {
                // 连续取空诊断：头部 chunk 与两侧队列状态（D 类问题定位用）。
                // 只嵌 rmtx_ 读队列长度；chunk/pend 字段按本文件既有调试
                // 打印惯例无锁读取（仅诊断，不保证精确快照）。
                static thread_local int fail_streak = 0;
                if (++fail_streak % 2000 == 0) {
                    std::size_t crdy, rdy;
                    {
                        std::lock_guard<std::mutex> lk(rmtx_);
                        crdy = cpu_ready_.size();
                        rdy = ready_.size();
                    }
                    fprintf(stderr,
                            "\n[pop-stall] side=%d crdy=%zu rdy=%zu "
                            "head=(side%d,%lld,end=%lld,exp=%lld,em=%lld) "
                            "pend=%d/%d\n",
                            (int)s, crdy, rdy,
                            (int)ch.side, (long long)ch.start_pts,
                            (long long)ch.end_pts, (long long)ch.expected,
                            (long long)ch.emitted, (int)side_pending_[0],
                            (int)side_pending_[1]);
                }
            }
            if (dbg) fprintf(stderr, "[hybrid-p] empty side=%d emitted_total=%lld\n", (int)s, (long long)emitted_total_);
            return false;
        }
        if (IsMarker(f)) {
            // 子解码器排空（EOF 后出现）
            if (s == SIDE_CPU && eof_pushed_) {
                // 全局 EOF：清空合并队列，marker 原样转发给调用方
                //（NextFrameImpl 的 kInt64 分支依赖它走 EOF 逻辑；后续
                // marker 继续透传）
                std::lock_guard<std::mutex> lk(mtx_);
                emit_queue_.pop_front();
                emit_queue_.clear();
                *frame = f;
                return true;
            }
            // GPU drain marker（或 CPU 侧未推流的 marker）：吞掉，
            // chunk 完成，继续下一 chunk
            std::lock_guard<std::mutex> lk(mtx_);
            emit_queue_.pop_front();
            continue;
        }
        if (f.pts < ch.start_pts) {
            // 陈旧帧（kick 边界帧在 expected 补齐后残留等）：丢弃
            std::lock_guard<std::mutex> lk(mtx_);
            --side_pending_[s];
            if (s == SIDE_GPU) --gpu_pending_;
            continue;
        }
        if (ch.end_pts != INT64_MAX && f.pts >= ch.end_pts) {
            // 越界帧（kick 产物 / 该侧更晚 chunk 的首帧）：stash 扣住，
            // 不关 chunk —— 本 chunk 的重排尾部帧（pts < end）可能仍在途。
            // 迟到帧继续从该侧产出、正常发射计数；expected 补满时在
            // 发射路径关闭 chunk。
            std::lock_guard<std::mutex> lk(mtx_);
            CHECK(!has_stash_[s]) << "stash overflow: side " << s;
            stash_[s] = f;
            has_stash_[s] = true;
            continue;
        }
        // 发射该帧（属于本 chunk）；expected 补满且非末 chunk 则关闭
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!emit_queue_.empty()) {
                Chunk &front = emit_queue_.front();
                ++front.emitted;
                static const bool no_exp = getenv("DECORD_HYBRID_NO_EXPECTED") != nullptr;
                if (!no_exp && front.expected > 0 && front.emitted >= front.expected
                        && front.end_pts != INT64_MAX) {
                    // 补满 expected 且非末 chunk：关闭。stash 里的越界帧
                    // 轮到该侧下一 chunk 时先出（pts 落在其区间或陈旧丢弃）
                    emit_queue_.pop_front();
                }
            }
            --side_pending_[s];
            if (s == SIDE_GPU) --gpu_pending_;
            ++emitted_total_;
        }
        frames_out_[s]++;
        *frame = f;
        return true;
    }
    // 合并队列已空：转发 CPU 侧 drain marker，让 VideoReader 的
    // EOF/rewind 逻辑（NextFrameImpl 的 kInt64 分支）正常运转
    // （GPU 驻留模式下 PopSide(SIDE_CPU) 读已上载队列，marker 由
    // UploadStep 直通）
    return PopSide(SIDE_CPU, frame);
}

runtime::NDArray HybridThreadedDecoder::ToHost(const runtime::NDArray &g) {
    CHECK(g.defined()) << "ToHost on undefined frame";
    const DLTensor *gtp = g.operator->();
    const DLTensor &gt = *gtp;
    runtime::NDArray h = runtime::NDArray::Empty(
        std::vector<int64_t>(gt.shape, gt.shape + gt.ndim), kUInt8, kCPU);
    h.pts = g.pts;
    // 三种输出格式 (rgb/gray/yuv420) 两侧布局逐字节一致：本 fork 的
    // yuv420 输出统一为 (h+ceil(h/2), w) 的 Y + 交错 UV 行（CPU filter 搬运
    // 与 GPU improc 直出的 packed 布局相同，实测逐字节相等）—— 无需重排，
    // 整块 D2H 即可。
    runtime::NDArray::CopyFromTo(const_cast<DLTensor *>(gtp),
                                 const_cast<DLTensor *>(h.operator->()), nullptr);
    return h;
}

#ifdef DECORD_USE_CUDA
bool HybridThreadedDecoder::LandStep() {
    if (!gpu_) return false;
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        if (ready_.size() >= ReadyCap()) return false;  // 预算内背压
    }
    runtime::NDArray f;
    if (!gpu_->Pop(&f) || !f.defined()) return false;
    // 落地速率（供调度）：段式统计 —— 只累计连续落地段（间隔 <50ms），
    // 段满 16 帧折算一次段速率并 EWMA。断流（chunk 间包断流/预算等待）
    // 重置段且不计入，否则"没活干"会被误判为"能力低"，调度锁死。
    // 必须在直通分支之前：GPU 驻留模式无 D2H，若统计挂在 ToHost 后
    // 则 gpu_rate_landed_ 恒 0 → "gpu-rate unknown" → CPU 包办
    // （实测 av1 f_c=0.8、混合退化）。
    {
        auto now = std::chrono::steady_clock::now();
        if (last_land_tp_.time_since_epoch().count() != 0) {
            double dt = std::chrono::duration<double>(now - last_land_tp_).count();
            if (dt > 1e-6 && dt < 0.05) {
                land_seg_frames_++;
                land_seg_secs_ += dt;
                if (land_seg_frames_ >= 16) {
                    double r = land_seg_frames_ / land_seg_secs_;
                    double prev = gpu_rate_landed_.load(std::memory_order_relaxed);
                    gpu_rate_landed_.store(
                        prev > 0 ? 0.5 * prev + 0.5 * r : r, std::memory_order_relaxed);
                    land_seg_frames_ = 0;
                    land_seg_secs_ = 0.0;
                }
            } else {
                land_seg_frames_ = 0;
                land_seg_secs_ = 0.0;
            }
        }
        last_land_tp_ = now;
    }
    if (IsMarker(f)) {
        // drain marker（kCPU kInt64）：在途 D2H 帧必须先于 marker 入队
        // （发射顺序 = ready_ 顺序）。ready_ 满时容忍越界（软限，仅
        // EOF 尾部发生一次）。
        FlushD2H();
        std::lock_guard<std::mutex> lk(rmtx_);
        ready_.push_back(std::move(f));
        return true;
    }
    if (out_cuda_) {
        // GPU 驻留模式：帧留在显存直通入队（零拷贝），随消费归还池
        std::lock_guard<std::mutex> lk(rmtx_);
        ready_.push_back(std::move(f));
        return true;
    }
    // 异步 D2H：提交到 pinned 中转环，滞后 kD2HRingSlots 帧收割
    // （事件早已完成，零等待）。GPU 源帧保活在环内，收割后回池。
    {
        const int k = static_cast<int>(d2h_seq_ & (kD2HRingSlots - 1));
        if (d2h_valid_[k] && !HarvestD2H(k)) return false;  // ready 满，稍后重试
        if (d2h_staging_[k] == nullptr
                || d2h_bytes_ != static_cast<std::size_t>(frame_bytes_)) {
            if (d2h_staging_[k] != nullptr) cudaFreeHost(d2h_staging_[k]);
            d2h_staging_[k] = nullptr;
            if (cudaMallocHost(&d2h_staging_[k],
                               static_cast<size_t>(frame_bytes_)) == cudaSuccess) {
                d2h_bytes_ = static_cast<std::size_t>(frame_bytes_);
            }
            if (d2h_ev_[k] == nullptr) {
                cudaEvent_t ev = nullptr;
                if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)
                        == cudaSuccess) {
                    d2h_ev_[k] = ev;
                }
            }
            if (d2h_stream_ == nullptr) {
                cudaStreamCreateWithFlags(
                    reinterpret_cast<cudaStream_t *>(&d2h_stream_),
                    cudaStreamNonBlocking);
            }
        }
        const char *src = static_cast<const char *>
            (const_cast<DLTensor *>(f.operator->())->data);
        if (d2h_staging_[k] != nullptr && d2h_ev_[k] != nullptr
                && d2h_stream_ != nullptr) {
            cudaMemcpyAsync(d2h_staging_[k], src,
                            static_cast<size_t>(frame_bytes_),
                            cudaMemcpyDeviceToHost,
                            reinterpret_cast<cudaStream_t>(d2h_stream_));
            cudaEventRecord(reinterpret_cast<cudaEvent_t>(d2h_ev_[k]),
                            reinterpret_cast<cudaStream_t>(d2h_stream_));
        } else {
            // pinned/event 分配失败：退回同步 D2H（必然完成）
            runtime::NDArray h = ToHost(f);
            std::lock_guard<std::mutex> lk(rmtx_);
            ready_.push_back(std::move(h));
            return true;
        }
        const DLTensor &ft = *f.operator->();
        d2h_host_[k] = runtime::NDArray::Empty(
            std::vector<int64_t>(ft.shape, ft.shape + ft.ndim), kUInt8, kCPU);
        d2h_host_[k].pts = f.pts;
        d2h_gpu_[k] = f;    // 源保活（D2H 完成前不回池）
        d2h_valid_[k] = true;
        ++d2h_seq_;
    }
    return true;
}

bool HybridThreadedDecoder::HarvestD2H(int k) {
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        if (ready_.size() >= ReadyCap()) return false;  // 预算内背压
    }
    cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(d2h_ev_[k]));
    std::memcpy(d2h_host_[k].operator->()->data, d2h_staging_[k],
                static_cast<size_t>(frame_bytes_));
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        ready_.push_back(std::move(d2h_host_[k]));
    }
    d2h_host_[k] = runtime::NDArray();
    d2h_gpu_[k] = runtime::NDArray();  // 源回池
    d2h_valid_[k] = false;
    return true;
}

void HybridThreadedDecoder::FlushD2H() {
    if (d2h_seq_ == 0) return;
    const int64_t base = d2h_seq_ - kD2HRingSlots;
    for (int64_t i = base; i < d2h_seq_; ++i) {
        if (i < 0) continue;
        const int k = static_cast<int>(i & (kD2HRingSlots - 1));
        if (!d2h_valid_[k]) continue;
        // marker 前的保序冲刷：ready_ 越界容忍（软限）；Stop 时放弃等待
        while (lander_run_.load() && !HarvestD2H(k)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!d2h_valid_[k]) continue;
        // 已停止：同步事件后丢弃（Clear/Stop 路径状态全清）
        cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(d2h_ev_[k]));
        d2h_host_[k] = runtime::NDArray();
        d2h_gpu_[k] = runtime::NDArray();
        d2h_valid_[k] = false;
    }
    d2h_seq_ = 0;
}

void HybridThreadedDecoder::AbortInflight() {
    // Clear/ROI 重建：同步在途拷贝（缓冲安全释放）后丢弃
    for (int k = 0; k < kD2HRingSlots; ++k) {
        if (d2h_valid_[k]) {
            if (d2h_ev_[k] != nullptr) {
                cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(d2h_ev_[k]));
            }
            d2h_host_[k] = runtime::NDArray();
            d2h_gpu_[k] = runtime::NDArray();
            d2h_valid_[k] = false;
        }
    }
    d2h_seq_ = 0;
    for (int k = 0; k < 4; ++k) {
        if (up_valid_[k]) {
            if (up_ev_[k] != nullptr) {
                cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(up_ev_[k]));
            }
            up_buf_[k] = runtime::NDArray();  // 显存缓冲随析构回 up_pool_
            up_valid_[k] = false;
        }
    }
}

bool HybridThreadedDecoder::FeedStep() {
    if (!gpu_) return false;
    bool has_pkt = false;
    ffmpeg::AVPacketPtr pkt;
    int flush_left = 0;
    {
        std::lock_guard<std::mutex> lk(lcv_mtx_);
        if (!gpu_pkt_q_.empty()) {
            pkt = std::move(gpu_pkt_q_.front());
            gpu_pkt_q_.pop_front();
            has_pkt = true;
        } else if (gpu_flush_left_ > 0) {
            flush_left = gpu_flush_left_;
        } else {
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        // ready_ 接近上限即暂停喂包，越界的包留在宿主队列（RAM）。
        // GPU 驻留模式：ready_ 帧本身持有池缓冲，物理上限 = 池深
        // （预留 NVDEC 在途 surface）；旧公式把整个池重复计入
        // （ready+pool ≥ ReadyCap → ready 到 ReadyCap-pool 即停喂），
        // CPU 块发射期 NVDEC 有效产出被压到 ~1200fps（hevc/h264 混跑
        // 的主要损耗源）。CPU-out：ready_ 是宿主帧，池缓冲仅在途
        // （D2H 环 + surface ≈ kGpuPoolBuffers），上限仍是 ReadyCap。
        {
            const std::size_t cap = out_cuda_
                ? static_cast<std::size_t>(std::max(gpu_pool_frames_, 1))
                : ReadyCap();
            const std::size_t reserve = out_cuda_
                ? static_cast<std::size_t>(
                      ThreadedDecoderInterface::kMaxOutputSurfaces + 8)
                : kGpuPoolBuffers;
            if (ready_.size() + reserve >= cap) {
                std::lock_guard<std::mutex> lk2(lcv_mtx_);
                if (has_pkt) gpu_pkt_q_.push_front(std::move(pkt));
                return false;
            }
        }
    }
    if (!has_pkt) {
        // EOF flush：每 surface 一个 flush 缓冲（与 VideoReader 的 GPU
        // EOF 语义一致，显示回调永远能取到缓冲而不阻塞）。部分推进安全：
        // 首个 null 即触发 ENDOFSTREAM，其余 buf 只是补足配对。
        int fed = 0;
        while (fed < flush_left) {
            runtime::NDArray buf;
            if (!gpu_pool_.Acquire(&buf)) break;  // 池耗尽，下轮再试
            gpu_->Push(nullptr, buf);
            ++fed;
        }
        std::lock_guard<std::mutex> lk(lcv_mtx_);
        gpu_flush_left_ = flush_left - fed;
        return fed > 0;
    }
    runtime::NDArray buf;
    if (!gpu_pool_.Acquire(&buf)) {
        // 池耗尽：包放回队首，等 LandStep 回收后重试（不阻塞工作线程）
        std::lock_guard<std::mutex> lk(lcv_mtx_);
        gpu_pkt_q_.push_front(std::move(pkt));
        return false;
    }
    gpu_->Push(std::move(pkt), buf);
    return true;
}

bool HybridThreadedDecoder::UploadStep() {
    if (!out_cuda_) return false;
    static const bool dbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
    runtime::NDArray buf;
    if (!up_pool_.Acquire(&buf)) {
        if (dbg) fprintf(stderr, "[hybrid-u] no-buf");
        return false;  // 上载池耗尽：CPU 帧显存容器已满（背压）
    }
    if (dbg) fprintf(stderr, "[hybrid-u] pop");
    runtime::NDArray f;
    if (!cpu_.Pop(&f) || !f.defined()) {
        if (dbg) fprintf(stderr, "[hybrid-u] cpu-empty");
        return false;  // buf 随析构归还池
    }
    if (dbg) fprintf(stderr, "[hybrid-u] copy");
    if (IsMarker(f)) {
        // marker 必须排在所有在途上载帧之后（发射顺序 = cpu_ready_ 顺序）
        FlushUpload();
        std::lock_guard<std::mutex> lk(rmtx_);
        cpu_ready_.push_back(std::move(f));
        return true;
    }
    // H2D：pinned 暂存环 + 非阻塞流。pageable 源的 H2D 在 WDDM 下走
    // 驱动 staging（~1.2ms/帧），pinned 源 ~0.4ms。提交后滞后 4 帧收割
    // （槽复用时事件早已完成）：每帧同步等待曾把 CPU chunk 发射压到
    // 实时跟随（上载 ~1000fps 上限）。两侧布局逐字节一致已验证。
    {
        const int k = up_stage_idx_ & 3;
        if (up_staging_[k] == nullptr || up_stage_bytes_ != static_cast<std::size_t>(frame_bytes_)) {
            if (up_staging_[k] != nullptr) {
                cudaFreeHost(up_staging_[k]);
                up_staging_[k] = nullptr;
            }
            if (cudaMallocHost(&up_staging_[k],
                               static_cast<size_t>(frame_bytes_)) == cudaSuccess) {
                up_stage_bytes_ = static_cast<std::size_t>(frame_bytes_);
            }
            if (up_ev_[k] == nullptr) {
                cudaEvent_t ev = nullptr;
                if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)
                        == cudaSuccess) {
                    up_ev_[k] = ev;
                }
            }
        }
        const char *src = static_cast<const char *>
            (const_cast<DLTensor *>(f.operator->())->data);
        char *dst_dev = static_cast<char *>
            (const_cast<DLTensor *>(buf.operator->())->data);
        if (up_staging_[k] != nullptr) {
            std::memcpy(up_staging_[k], src, static_cast<size_t>(frame_bytes_));
            cudaMemcpyAsync(dst_dev, up_staging_[k],
                            static_cast<size_t>(frame_bytes_),
                            cudaMemcpyHostToDevice,
                            reinterpret_cast<cudaStream_t>(up_stream_));
        } else {
            cudaMemcpyAsync(dst_dev, src, static_cast<size_t>(frame_bytes_),
                            cudaMemcpyHostToDevice,
                            reinterpret_cast<cudaStream_t>(up_stream_));
        }
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(up_stream_));
        ++up_stage_idx_;
    }
    buf.pts = f.pts;
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        cpu_ready_.push_back(std::move(buf));
    }
    if (dbg) fprintf(stderr, "[hybrid-u] done");
    return true;
}

void HybridThreadedDecoder::FlushUpload() {
    // marker 保序：按提交顺序同步事件后入队（EOF 尾部一次性，容忍等待）
    const int64_t in_use = std::min<int64_t>(4, up_stage_idx_);
    for (int64_t i = in_use; i > 0; --i) {
        const int k = static_cast<int>((up_stage_idx_ - i) & 3);
        if (!up_valid_[k]) continue;
        if (up_ev_[k] != nullptr) {
            cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(up_ev_[k]));
        }
        {
            std::lock_guard<std::mutex> lk(rmtx_);
            cpu_ready_.push_back(std::move(up_buf_[k]));
        }
        up_valid_[k] = false;
    }
}

void HybridThreadedDecoder::GpuWorkerLoop() {
    static const bool dbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
    while (lander_run_.load()) {
        bool did = LandStep();
        did = FeedStep() || did;
        if (dbg && did) fprintf(stderr, "[hybrid-w] did=%d ready=%zu cpuready=%zu pktq=%zu\n", (int)did, ready_.size(), cpu_ready_.size(), gpu_pkt_q_.size());
        if (!did) {
            if (dbg) fprintf(stderr, "[hybrid-w] idle rdy=%zu crdy=%zu q=%zu",
                             ready_.size(), cpu_ready_.size(), gpu_pkt_q_.size());
            std::unique_lock<std::mutex> lk(lcv_mtx_);
            lcv_.wait_for(lk, std::chrono::milliseconds(1));
        }
    }
}

void HybridThreadedDecoder::UploaderLoop() {
    while (lander_run_.load()) {
        if (!UploadStep()) {
            std::unique_lock<std::mutex> lk(lcv_mtx_);
            lcv_.wait_for(lk, std::chrono::milliseconds(1));
        }
    }
}
#endif  // DECORD_USE_CUDA

bool HybridThreadedDecoder::Drained() const {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!emit_queue_.empty() || has_stash_[0] || has_stash_[1]) return false;
    }
#ifdef DECORD_USE_CUDA
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        if (!ready_.empty()) return false;
        if (out_cuda_ && !cpu_ready_.empty()) return false;
    }
    {
        std::lock_guard<std::mutex> lk(lcv_mtx_);
        if (!gpu_pkt_q_.empty() || gpu_flush_left_ > 0) return false;
    }
#endif
    if (!cpu_.Drained()) return false;
#ifdef DECORD_USE_CUDA
    if (gpu_ && !gpu_->Drained()) return false;
#endif
    return true;
}

void HybridThreadedDecoder::SuggestDiscardPTS(std::vector<int64_t> dts) {
    cpu_.SuggestDiscardPTS(dts);
#ifdef DECORD_USE_CUDA
    if (gpu_) gpu_->SuggestDiscardPTS(dts);
#endif
}

void HybridThreadedDecoder::ClearDiscardPTS() {
    cpu_.ClearDiscardPTS();
#ifdef DECORD_USE_CUDA
    if (gpu_) gpu_->ClearDiscardPTS();
#endif
}

}  // namespace decord
