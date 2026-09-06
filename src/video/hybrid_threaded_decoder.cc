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
/*! 冷启动阶段交替分配（前两个 chunk 各来一次，让两路速率都可测） */
constexpr int kColdStartChunks = 2;
constexpr double kRateEwmaAlpha = 0.5;
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
    // （一次性 cudaMalloc 数百块的首帧开销实测 ≈15% 吞吐）
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
        vram_budget = static_cast<double>(free_b) * 0.45;
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
    prefetch_frames_ = clampi(queue_frames_ * 2, 192, 1536);
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
        // GPU 输出缓冲池与 ready_ 预算换算基准
        gpu_frame_shape_ = GpuFrameShape();
        frame_bytes_ = 1;  // kUInt8
        for (int64_t d : gpu_frame_shape_) frame_bytes_ *= d;
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
        // 此刻无在途缓冲（上面守卫），Reset 干净。
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
    backlog_end_[0] = backlog_end_[1] = 0;
    emitted_upto_[0] = emitted_upto_[1] = 0;
    chunks_assigned_[0] = chunks_assigned_[1] = 0;
    emitted_total_ = 0;
    side_pending_[0] = side_pending_[1] = 0;
    last_chunk_frames_ = 0;
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
    // rate_ewma_ / kf 索引保留：Seek 后复用学到的两侧速率与帧数表
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
    // AV1 与 IDR 型走统一 min-max 贪心：实测 racelog AV1 流 has_b_frames=0
    // 且包 pts==dts（解码序=显示序），跨侧切换无重排残留、不需要 kick
    // （kick 的 show_existing 映射反而造成双发错位）。重排流的安全网见
    // Pop 的 force-close。
    // 冷启动：前 kColdStartChunks 个 chunk 交替分配，采集两侧速率
    int total = chunks_assigned_[0] + chunks_assigned_[1];
    if (total < kColdStartChunks) {
        return total == 0 ? SIDE_CPU : SIDE_GPU;
    }
    // 硬预算：GPU 待发射帧字节数超限即强制 CPU 承接。这是调度失衡时
    // GPU 包队列/在途帧的总量上限（gpu_pending_ 按包粒度精确计数）。
    if (frame_bytes_ > 0
            && gpu_pending_ * frame_bytes_ > kGpuAheadBytes) {
        DLOG(INFO) << "[hybrid/sched] key=" << key_pts << " budget-exceeded pending=" << gpu_pending_ << " -> CPU";
        return SIDE_CPU;
    }
    // 积压排空贪心（帧数口径）：t[s] = (该侧已分配未发射帧数 + 新 chunk
    // 估计帧数) / 该侧速率(帧/秒)，给 t 小的侧。与"上次发射位置"无关
    // —— 用发射位置做距离会让正在发射的一侧 dist 恒小、调度自锁
    //（实测 GPU 被锁死在冷启动 chunk，util 17%）。
    // CPU 速率取发射侧 EWMA（积压的真实释放速率，min-max 的正确量纲）；
    // GPU 用落地段速率。未学得时回退生产侧实测（解码能力上限）。
    double r_cpu = rate_ewma_[SIDE_CPU] > 0
                       ? rate_ewma_[SIDE_CPU]
                       : cpu_.ProductionRate();
    double rate[2] = {r_cpu,
                      gpu_rate_landed_.load(std::memory_order_relaxed)};
    int64_t backlog[2] = {side_pending_[SIDE_CPU], side_pending_[SIDE_GPU]};
    // 积压钳制：某侧失去 chunk 分配后，其积压只在"front 轮到该侧"时排空
    // —— 不再分 chunk 就永不排空，min-max 的完工时间被幻觉积压顶死
    //（实测 av1 小 GOP：GPU pending 600 恒挂、f_c 漂到 0.97）。按 chunk
    // 尺寸钳制上限，保证落后侧能重新赢得分配、积压真实流动。
    {
        const int64_t cap = 2 * last_chunk_frames_ + 64;
        backlog[0] = std::min(backlog[0], cap);
        backlog[1] = std::min(backlog[1], cap);
    }
    // 速率未测得（测量天然滞后于分配）：不给 CPU。消费驱动 demux 下
    // CPU 包到达即发射前夕（仅 prefetch 领先），分给 CPU 的 chunk 必然
    // 以实时解码速率发射 —— 未证明有益（rate_cpu >= 已知吞吐）的 CPU
    // 份额只会拖慢整体。GPU 侧兜底（解码快到能自补，且 GPU 包经宿主
    // 队列与消费解耦）。冷启动的 chunk2 强制 GPU 保证两侧速率都被测到。
    if (rate[SIDE_CPU] <= 0) {
        DLOG(INFO) << "[hybrid/sched] cpu-rate unknown -> GPU";
        return SIDE_GPU;
    }
    if (rate[SIDE_GPU] <= 0) {
        DLOG(INFO) << "[hybrid/sched] gpu-rate unknown -> CPU";
        return SIDE_CPU;
    }
    // 慢侧弃用闸：CPU 包到达即发射前夕（消费驱动 demux），无法离峰生产，
    // 分给 CPU 的 chunk 以实时解码速率发射 —— 当 CPU 实测速率低于 GPU 时，
    // 任何 CPU 份额都是净拖累（test6_hevc 长视频实测 0.88x：water-filling
    // 仍按比例给 CPU 26% 份额）。两个速率都是产出点实测（软解 filter 线程
    // / NVDEC 落地段），口径可比。h264（软解 ~1200 > NVDEC ~975）不受影响，
    // 互补保留（长视频实测 1.77x）。
    if (rate[SIDE_CPU] < rate[SIDE_GPU]) {
        return SIDE_GPU;
    }
    // 份额均衡（water-filling 计数器）：目标份额 ∝ 实测速率，按"实际分配
    // 帧数与理想份额的差"决策。deficit 精确、无 tie 陷阱 —— min-max 在
    // 积压钳制成对称后两侧 work 恒等、t 恒 tie，tie->CPU 使 f_c 漂移到 1
    //（test6 全量实测 6000 帧后断崖跌到 529fps、CPU 独占、GPU 闲置）。
    // 积压安全由 GPU 字节预算 + 池耗尽背压独立兜底。
    const double est = static_cast<double>(est_chunk_frames_);
    const double tot = rate[SIDE_CPU] + rate[SIDE_GPU];
    const double want_cpu = (static_cast<double>(alloc_frames_[0]
                            + alloc_frames_[1]) + est) * rate[SIDE_CPU] / tot;
    Side chosen = (static_cast<double>(alloc_frames_[SIDE_CPU]) <= want_cpu)
                      ? SIDE_CPU : SIDE_GPU;
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
                Chunk last{cur_side_, cur_start_pts_, INT64_MAX, 0, 0, {}, false};
                emit_queue_.push_back(last);
                backlog_end_[cur_side_] = INT64_MAX;

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
            // 首包：首 chunk 固定 CPU（首帧延迟优先 + CPU 速率学习）。
            // 跨侧冲刷仅 IDR 型需要 kick；AV1 无重排流（pts==dts）不依赖
            // kick，混合路由由 Pop 的 force-close 安全网兜底。
            routing_active_ = true;
            cur_side_ = SIDE_CPU;
            cur_start_pts_ = pkt->pts;
            Chunk c{cur_side_, pkt->pts, INT64_MAX, 0, 0, {}, false};
            emit_queue_.push_back(c);
            ++chunks_assigned_[cur_side_];
            backlog_end_[SIDE_CPU] = INT64_MAX;
        } else if (is_key) {
            // 关闭当前 chunk（end = 本关键帧 pts），为新 chunk 选侧
            if (!emit_queue_.empty()) {
                Chunk &last = emit_queue_.back();
                last.end_pts = pkt->pts;
                last.expected = ExpectedFrames(last.start_pts, pkt->pts);
                if (last.expected > 0) {
                    last_chunk_frames_ = last.expected;
                    est_chunk_frames_ = last.expected;
                }
            }
            Side old_side = cur_side_;
            Side s = ChooseSide(pkt->pts);
            // kick 冲刷仅 IDR 型：其关键帧重置参考状态、迫使重排帧交出，
            // 且输出 pts == 包 pts（stash 闭环成立）。AV1 的 show_existing
            // 映射使 kick 帧输出 pts 回落上一 chunk（实测双发错位），且
            // 无重排流（pts==dts）边界本就干净 —— 跨侧直接切换不发 kick，
            // 重排流的残留由 Pop 的 force-close 兜底。
            if (s != old_side && IsIdrLikeCodec()) {
                flush_side = old_side;
                need_flush = true;
            }
            cur_side_ = s;
            cur_start_pts_ = pkt->pts;
            Chunk c{s, pkt->pts, INT64_MAX, 0, 0, {}, false};
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

void HybridThreadedDecoder::RecordChunkRate(const Chunk &ch) {
    // 速率统一为 帧/秒（与 GPU 落地速率同单位；dist 也换算成帧距，
    // 否则跨单位比较会让 GPU 的预计排空时间虚大数倍、永远输给 CPU）
    if (ch.timed) {
        double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - ch.first_pop)
                        .count();
        double frames = 0.0;
        if (ch.expected > 0) {
            frames = static_cast<double>(ch.expected);
        } else if (pts_per_frame_ > 0) {
            frames = static_cast<double>(ch.end_pts - ch.start_pts) / pts_per_frame_;
        }
        if (dt > 1e-6 && frames > 0) {
            double inst = frames / dt;
            rate_ewma_[ch.side] =
                rate_ewma_[ch.side] > 0
                    ? (1.0 - kRateEwmaAlpha) * rate_ewma_[ch.side] + kRateEwmaAlpha * inst
                    : inst;
        }
    }
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
                    RecordChunkRate(emit_queue_.front());
                    emit_queue_.pop_front();
                    close_it = true;
                }
            }
            if (close_it) {
                DLOG(INFO) << "[hybrid] force-close chunk start=" << ch.start_pts
                           << " emitted=" << ch.emitted << "/" << ch.expected;
                continue;
            }
        }
        Side s = ch.side;
        runtime::NDArray f;
        if (!PopSide(s, &f)) {
            static const bool dbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
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
                RecordChunkRate(emit_queue_.front());
                emit_queue_.pop_front();
                emit_queue_.clear();
                *frame = f;
                return true;
            }
            // GPU drain marker（或 CPU 侧未推流的 marker）：吞掉，
            // chunk 完成，继续下一 chunk
            std::lock_guard<std::mutex> lk(mtx_);
            RecordChunkRate(emit_queue_.front());
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
                if (!front.timed) {
                    front.timed = true;
                    front.first_pop = std::chrono::steady_clock::now();
                }
                ++front.emitted;
                static const bool no_exp = getenv("DECORD_HYBRID_NO_EXPECTED") != nullptr;
                if (!no_exp && front.expected > 0 && front.emitted >= front.expected
                        && front.end_pts != INT64_MAX) {
                    // 补满 expected 且非末 chunk：关闭。stash 里的越界帧
                    // 轮到该侧下一 chunk 时先出（pts 落在其区间或陈旧丢弃）
                    RecordChunkRate(front);
                    emit_queue_.pop_front();
                }
            }
            --side_pending_[s];
            if (s == SIDE_GPU) --gpu_pending_;
            emitted_upto_[s] = f.pts;
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
        // drain marker（kCPU kInt64）：直接透传，不占字节预算的实质空间
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
    // 同步 D2H 到独立宿主缓冲；GPU 缓冲随 f 析构回收进有界池
    runtime::NDArray h = ToHost(f);
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        ready_.push_back(std::move(h));
    }
    return true;
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
        // ready_ 接近预算即暂停喂包：GPU 解码超前被限制在
        // ready_ 余量 + 池缓冲内，越界的包留在宿主队列（RAM）
        if (ready_.size() + kGpuPoolBuffers >= ReadyCap()) {
            std::lock_guard<std::mutex> lk2(lcv_mtx_);
            if (has_pkt) gpu_pkt_q_.push_front(std::move(pkt));
            return false;
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
        std::lock_guard<std::mutex> lk(rmtx_);
        cpu_ready_.push_back(std::move(f));
        return true;
    }
    // H2D：pinned 暂存环 + 非阻塞流。pageable 源的 H2D 在 WDDM 下走
    // 驱动 staging（~1.2ms/帧），pinned 源 ~0.4ms；本帧提交后 sync（环
    // 槽在 4 帧后才复用，sync 语义安全）。两侧布局逐字节一致已验证。
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
    if (dbg) fprintf(stderr, "[hybrid-u] push");
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        cpu_ready_.push_back(std::move(buf));
    }
    if (dbg) fprintf(stderr, "[hybrid-u] done");
    return true;
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
