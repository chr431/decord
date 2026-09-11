/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file threaded_decoder.cc
 * \brief FFmpeg threaded decoder Impl
 */

#include "threaded_decoder.h"

#include <dmlc/logging.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <chrono>
#include "../../runtime/str_util.h"
#include <iostream>
// TEMP PROFILE
static const bool DECORD_PROFILE = std::stoi(
    decord::runtime::GetEnvironmentVariableOrDefault("DECORD_PROFILE", "0")) != 0;
struct _PfAcc2 {
    std::chrono::steady_clock::time_point t0; double acc = 0.0; long long n = 0;
    void start() { if (DECORD_PROFILE) t0 = std::chrono::steady_clock::now(); }
    void stop() { if (!DECORD_PROFILE) return;
        acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); ++n; }
};
static _PfAcc2 pf_d_send, pf_d_recv, pf_d_push, pf_f_pop, pf_f_filter, pf_f_push;
#include <iostream>



namespace decord {
namespace ffmpeg {

// Upper bound on frame_queue_ to prevent unbounded memory growth.
// When the consumer (Python) is slower than the producer (decoder),
// the queue would otherwise grow to hold every decoded frame.
// 0 = unlimited (old behaviour).
static const int DECORD_CPU_FRAME_QUEUE_SIZE = std::stoi(
    decord::runtime::GetEnvironmentVariableOrDefault("DECORD_CPU_FRAME_QUEUE_SIZE", "32"));

// EAGAIN yield interval in ms (experimental tuning).
static const int DECORD_EAGAIN_SLEEP_MS = std::stoi(
    decord::runtime::GetEnvironmentVariableOrDefault("DECORD_EAGAIN_SLEEP_MS", "1"));

// FFmpeg 8+: use synchronous decode mode (AV_CODEC_RECEIVE_FRAME_FLAG_SYNCHRONOUS)
// which bypasses internal frame threading.  This eliminates EAGAIN from
// avcodec_send_packet and reduces thread-synchronisation overhead, which can
// help latency-sensitive workloads.  Frame-threading generally wins for
// throughput, so this defaults to OFF.  Set DECORD_SYNC_DECODE=1 to enable.
static const bool DECORD_SYNC_DECODE = std::stoi(
    decord::runtime::GetEnvironmentVariableOrDefault("DECORD_SYNC_DECODE", "0")) != 0;

// Decode-side backpressure slack: in-flight frames the decode thread may
// keep ahead of the consumer beyond the (bounded) output frame queue.
// EnqueueRawFrame waits once raw_queue_ + frame_queue_ reach
// max_queue_frames_ + this.  Without it the raw (pre-filter) queue is
// unbounded and a contested filter (e.g. two decoders running concurrently)
// lets raw full-frames accumulate without limit (observed ~1.9 MB/frame
// runaway to multi-GB on a 1080p VFR clip pair).
static const int DECORD_RAW_SLACK_FRAMES = 4;

// FFmpeg 8+: prefer avcodec_receive_frame_flags with SYNCHRONOUS flag
// to bypass internal frame threading overhead.  Falls back to the
// standard avcodec_receive_frame on older FFmpeg or when sync decode
// is disabled via DECORD_SYNC_DECODE=0.
static inline int ReceiveFrame(AVCodecContext *ctx, AVFrame *frame) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(62, 0, 0)
    if (DECORD_SYNC_DECODE) {
        return avcodec_receive_frame_flags(
            ctx, frame, AV_CODEC_RECEIVE_FRAME_FLAG_SYNCHRONOUS);
    }
#endif
    return avcodec_receive_frame(ctx, frame);
}

FFMPEGThreadedDecoder::FFMPEGThreadedDecoder()
    : frame_count_(0), draining_(false), run_(false),
      error_status_(false), error_message_(),
      max_queue_frames_(DECORD_CPU_FRAME_QUEUE_SIZE),
      codec_is_av1_(false) {
}

void FFMPEGThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, int width, int height, int rotation, int output_format) {
    bool running = run_.load();
    Clear();
    dec_ctx_.reset(dec_ctx);
    orig_w_ = dec_ctx->width;
    orig_h_ = dec_ctx->height;
    out_w_ = width;
    out_h_ = height;
    rotation_ = rotation;
    output_format_ = output_format;
    // 转换扇出池：默认关闭（单线程旧路径）。实测 RGB 吞吐瓶颈在 get_batch
    // 的批组装拷贝与解码，转换已被解码时间掩盖（workers 1/4/8 均为
    // ~810-840fps，无扩展）—— Nelux 的转换扇出收益来自其零拷贝 tensor
    // 迭代消费，与我们的批组装 API 不同。池保留给 4K/缩放输出等转换
    // 主导场景：DECORD_CONVERT_WORKERS=N 显式启用。yuv420/gray 恒单线程。
    convert_workers_ = 1;
    if (output_format_ == 0) {
        if (const char *e = getenv("DECORD_CONVERT_WORKERS"))
            convert_workers_ = std::max(1, atoi(e));
    }
    color_range_ = (dec_ctx->color_range == AVCOL_RANGE_JPEG) ? 1 : 0;
    // AV1（dav1d）批量解码模式标记：dav1d 帧并行需要多 packet 在途
    codec_is_av1_ = (dec_ctx->codec_id == AV_CODEC_ID_AV1);
    // avcodec_flush_buffers（Seek/Clear 触发）会把 time_base 重置为
    // 0/1 —— filter 图 buffersrc 需要流 time_base（Invalid time base
    // 0/1）。在此快照，重建图前恢复。
    time_base_ = dec_ctx->time_base;
    {
        std::lock_guard<std::mutex> lk(filter_mutex_);
        BuildFilterGraph();
    }
    if (running) {
        Start();
    }
}

void FFMPEGThreadedDecoder::SetRoi(int x1, int y1, int x2, int y2) {
    int w = x2 - x1;
    int h = y2 - y1;
    bool valid = w > 0 && h > 0 && x1 >= 0 && y1 >= 0
                 && x2 <= orig_w_ && y2 <= orig_h_;
    if (valid) {
        roi_x1_ = x1; roi_y1_ = y1; roi_x2_ = x2; roi_y2_ = y2;
        roi_valid_ = true;
    } else {
        roi_x2_ = -1; roi_y2_ = -1;
        roi_valid_ = false;
    }
    // 热切换 filter 图：不停止工作线程、不 flush avcodec、不丢在途帧。
    // 在途帧仍走旧图（全帧输出），新帧走新图（ROI 输出）——消费者端
    // CropRoi 按帧实际尺寸自适应（ROI 尺寸直通，全帧尺寸旧路径裁剪）。
    // 注意：SetRoi 前已经由旧图输出的帧会被消费方正确裁剪，无帧序影响。
    std::lock_guard<std::mutex> lk(filter_mutex_);
    BuildFilterGraph();
}

void FFMPEGThreadedDecoder::BuildFilterGraph() {
    // ── filter 描述串 ──
    // 强制 BT.601 (limited) 颜色转换：CUDA 解码路径用固定 BT.601 矩阵
    // （improc.cu），而 FFmpeg 的 swscale 遵循流的颜色元数据（多数赛车
    // 视频标注 bt709）——两者对同一帧产生可见差异的 RGB，破坏 CPU 路径
    // OCR（实测 28/30 识别失败、G 通道系统性 +7.5 偏移）。setparams 重写
    // 帧元数据让 swscale 选 BT.601 矩阵，输出与 GPU 路径对齐（逐像素差
    // ≤1-2）。
    //
    // ROI-first（性能）：roi_valid 且 gray/nv12 输出、无旋转、无用户缩放时，
    // crop 先于 format —— 只转换 ROI 像素。gray 只取 luma：crop 对 luma
    // 平面按任意坐标精确裁剪；nv12 的色度按 2x2 块裁剪（VideoReader::SetRoi
    // 已把 ROI 扩成偶数超集，CropRoi 再精裁回精确矩形）。RGB 输出不启用
    // （色度上采样在裁剪边界需要窗口外的色度样本，与旧全帧转换逐像素
    // 不一致），保持旧路径。
    // 旋转非 0 或用户缩放时同样保持旧行为（全帧转换 + 调用方裁剪）。
    // yuv420 输出直出 NV12（sws SIMD 交错）而非 yuv420p：输出布局同为
    // packed 2D（Y 行 + 交错 UV 行），但 CopyToNDArray 的打包从"逐字节
    // U/V 交错标量循环"（~518K 次/帧@1080p，是 CPU 侧有效供给的主成本）
    // 变成两次整块 memcpy —— 软解侧供给 800 → 1000+fps（hybrid 调度的
    // 速率口径与慢侧弃用闸均按 filter 后口径）。
    const char *fmt = output_format_ == 2 ? "nv12"
                      : (output_format_ == 1 ? "gray" : "rgb24");
    bool user_scale = (out_w_ > 0 && out_h_ > 0
                       && (out_w_ != orig_w_ || out_h_ != orig_h_));
    bool crop_first = roi_valid_ && rotation_ == 0 && !user_scale
                      && output_format_ != 0;
    int crop_w = roi_x2_ - roi_x1_;
    int crop_h = roi_y2_ - roi_y1_;
    char descr[256];    switch (rotation_) {
        case 90:
            std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,transpose=1,scale=%d:%d,format=%s", out_w_, out_h_, fmt);
            break;
        case 180:
            std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,transpose=1,transpose=1,scale=%d:%d,format=%s", out_w_, out_h_, fmt);
            break;
        case 270:
            std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,transpose=2,scale=%d:%d,format=%s", out_w_, out_h_, fmt);
            break;
        case 0:
        default:
            if (crop_first) {
                std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,crop=%d:%d:%d:%d,format=%s", crop_w, crop_h, roi_x1_, roi_y1_, fmt);
            } else if (user_scale) {
                std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,scale=%d:%d,format=%s", out_w_, out_h_, fmt);
            } else {
                std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,format=%s", fmt);
            }
    }
    // 恢复流 time_base（Clear/flush 会重置为 0/1，buffersrc 需要有效值）
    if (dec_ctx_.get()) {
        dec_ctx_->time_base = time_base_;
    }
    filter_graph_ = FFMPEGFilterGraphPtr(new FFMPEGFilterGraph(descr, dec_ctx_.get(), output_format_));
    // 扇出池：快照描述串并推进代数 —— 各 worker 在处理下一帧时按代数
    // 重建自己的本地 graph（SetRoi 热切换对池同样生效）。
    graph_descr_ = descr;
    ++graph_gen_;
}

void FFMPEGThreadedDecoder::Start() {
    CheckErrorStatus();
    if (!run_.load()) {
        pkt_queue_.reset(new PacketQueue());
        raw_queue_.reset(new RawFrameQueue());
        frame_queue_.reset(new FrameQueue());
        buffer_queue_.reset(new BufferQueue());
        run_.store(true);
        auto t = std::thread(&FFMPEGThreadedDecoder::WorkerThread, this);
        std::swap(t_, t);
        if (convert_workers_ > 1) {
            StartConvertPool();
        } else {
            auto ft = std::thread(&FFMPEGThreadedDecoder::FilterWorkerThread, this);
            std::swap(filter_t_, ft);
        }
    }
}

void FFMPEGThreadedDecoder::Stop() {
    if (run_.load()) {
        if (pkt_queue_) {
            pkt_queue_->SignalForKill();
        }
        if (raw_queue_) {
            raw_queue_->SignalForKill();
        }
        if (buffer_queue_) {
            buffer_queue_->SignalForKill();
        }
        run_.store(false);
        if (frame_queue_) {
            frame_queue_->SignalForKill();
        }
        bp_cv_.notify_all();  // 唤醒仍等背压的解码/转换线程
    }
    if (t_.joinable()) {
        // LOG(INFO) << "joining";
        t_.join();
    }
    if (filter_t_.joinable()) {
        filter_t_.join();
    }
    StopConvertPool();
}

void FFMPEGThreadedDecoder::StartConvertPool() {
    StopConvertPool();  // 防御：残留线程先清
    for (int i = 0; i < convert_workers_; ++i) {
        convert_worker_threads_.emplace_back(&FFMPEGThreadedDecoder::ConvertWorkerLoop, this);
    }
}

void FFMPEGThreadedDecoder::StopConvertPool() {
    if (!convert_worker_threads_.empty()) {
        for (auto &w : convert_worker_threads_) {
            if (w.joinable()) w.join();
        }
        convert_worker_threads_.clear();
    }
    std::lock_guard<std::mutex> lk(reorder_mu_);
    reorder_map_.clear();
    reorder_next_ = 0;
    raw_seq_ = 0;
}

void FFMPEGThreadedDecoder::Clear() {
    Stop();
    if (dec_ctx_.get()) {
        avcodec_flush_buffers(dec_ctx_.get());
    }
    frame_count_.store(0);
    draining_.store(false);
    {
      std::lock_guard<std::mutex> lock(pts_mutex_);
      discard_pts_.clear();
    }
    error_status_.store(false);
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_.clear();
    }
}

void FFMPEGThreadedDecoder::SuggestDiscardPTS(std::vector<int64_t> dts) {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    discard_pts_.insert(dts.begin(), dts.end());
}

void FFMPEGThreadedDecoder::ClearDiscardPTS() {
    std::lock_guard<std::mutex> lock(pts_mutex_);
    discard_pts_.clear();
}

void FFMPEGThreadedDecoder::Push(AVPacketPtr pkt, runtime::NDArray buf) {
    CheckErrorStatus();
    if (!run_.load()) {
        LOG(FATAL) << "Push() called after the decoder worker stopped "
                   << "(decoder thread exited unexpectedly). Check the previous "
                   << "decoder error before reading more frames.";
    }
    if (!pkt) {
        CHECK(!draining_.load()) << "Start draining twice...";
        draining_.store(true);
    }

    pkt_queue_->Push(pkt);
    buffer_queue_->Push(buf);

    // LOG(INFO)<< "frame push: " << frame_count_;
    // LOG(INFO) << "Pushed pkt to pkt_queue";
}

bool FFMPEGThreadedDecoder::Pop(runtime::NDArray *frame) {
    // Pop is blocking operation
    // unblock and return false if queue has been destroyed.

    CheckErrorStatus();
    if (!frame_count_.load() && !draining_.load()) {
        return false;
    }
    // LOG(INFO) << "Waiting for pop";
    bool ret = frame_queue_->Pop(frame);
    // LOG(INFO) << "Poped";
    CheckErrorStatus();

    if (ret) {
        --frame_count_;
        // 消费腾出槽位：唤醒等背压的解码线程（EnqueueRawFrame）
        bp_cv_.notify_all();
    }
    return (ret && frame->data_);
}

bool FFMPEGThreadedDecoder::Drained() const {
    return !frame_count_.load() && !draining_.load();
}

FFMPEGThreadedDecoder::~FFMPEGThreadedDecoder() {
    Stop();
}

void FFMPEGThreadedDecoder::ProcessFrame(AVFramePtr frame, NDArray out_buf) {
    // filter image frame (format conversion, scaling...) — runs on the
    // filter worker thread, concurrent with the decode worker.
    // 每帧在锁内拷贝当前 filter 图指针：SetRoi 的热切换不会让在途帧
    // 引用到被销毁的旧图（shared_ptr 保活到本帧处理完成）。
    std::shared_ptr<FFMPEGFilterGraph> graph;
    {
        std::lock_guard<std::mutex> lk(filter_mutex_);
        graph = filter_graph_;
    }
    pf_f_filter.start();
    graph->Push(frame.get());
    // 生产速率段式 EWMA（混合解码调度用）：统计 filter 后的帧产出
    // 节奏（NV12 直出后 filter 成本 ≈ 解码成本，产出口径即有效供给）。
    // 注意不可按存货水位拒计样本 —— GPU-first 冷启动下 CPU 解码器
    // 恒处深存货状态（包永远充足），水位门控会把 rc 永久屏蔽
    // （实测 hevc 16 chunks 全部 rc-unknown，退化为 CPU 主导 667fps）。
    {
        auto now = std::chrono::steady_clock::now();
        static const bool rdbg = getenv("DECORD_CPU_RATE_DEBUG") != nullptr;
        if (last_prod_tp_.time_since_epoch().count() != 0) {
            double dt = std::chrono::duration<double>(now - last_prod_tp_).count();
            if (dt > 0.05) {
                if (rdbg) fprintf(stderr, "[rate] RESET dt=%.0fms seg=%lld\n",
                                  dt * 1000, (long long)prod_seg_frames_);
                prod_seg_frames_ = 0;
                prod_seg_secs_ = 0.0;
            } else if (dt > 1e-6) {
                prod_seg_frames_++;
                prod_seg_secs_ += dt;
                if (prod_seg_frames_ >= 16) {
                    const double r = prod_seg_frames_ / prod_seg_secs_;
                    ++prod_folds_;  // relaxed 自增：int64 计数，仅诊断/门控用
                    prod_fold_ring_[prod_fold_i_] = r;
                    prod_fold_i_ = (prod_fold_i_ + 1) % kProdFoldRing;
                    if (prod_fold_n_ < kProdFoldRing) ++prod_fold_n_;
                    // prod_rate_ = 滑窗持续产能 + 慢降锁（唯一口径，2026-09-12
                    // 起替代容量跟踪 EWMA）：max(最近8折内"连续4折最小值"的
                    // 最大值, 上一值×0.997)。排除解码器预跑存货的排空孤峰
                    // （撑不满连续 4 折；旧 EWMA 被逐发 1.3x 棘轮永久卡
                    // 1200-2500，hevc 真值 ≈754），慢降锁保真实产能穿越饿供
                    // 相位（h264 纯臂 3573 只在整包喂足段显形）。≥4 折才出版
                    // （否则 0 = 未学得，调度走 CPU-sample chunk 冷启动）。
                    // 混合调度两条路径（cpu-out/gpu-out）统一采用：gpu-out 的
                    // 倾斜计划死锁已由 kick 突发治本（.h KICK_BURST），实测
                    // hevc 引擎全片 e2e −24%（见 .h 注释数字表）。
                    if (prod_fold_n_ >= 4) {
                        double best = 0.0;
                        const int oldest = (prod_fold_i_ - prod_fold_n_
                                            + 2 * kProdFoldRing) % kProdFoldRing;
                        for (int i = 0; i + 4 <= prod_fold_n_; ++i) {
                            double w = prod_fold_ring_[(oldest + i) % kProdFoldRing];
                            for (int j = 1; j < 4; ++j) {
                                const double v = prod_fold_ring_
                                    [(oldest + i + j) % kProdFoldRing];
                                if (v < w) w = v;
                            }
                            if (w > best) best = w;
                        }
                        const double prev = prod_rate_.load(std::memory_order_relaxed);
                        prod_rate_.store(
                            prev > 0 ? std::max(best, prev * 0.997) : best,
                            std::memory_order_relaxed);
                    }
                    if (rdbg) fprintf(stderr,
                                      "[rate] FOLD t=%.3f r=%.0f rate=%.0f n=%d\n",
                                      std::chrono::duration<double>(
                                          std::chrono::steady_clock::now().time_since_epoch()).count(),
                                      r, prod_rate_.load(), prod_fold_n_);
                    prod_seg_frames_ = 0;
                    prod_seg_secs_ = 0.0;
                }
            }
        }
        last_prod_tp_ = now;
    }
    AVFramePtr out_frame = AVFramePool::Get()->Acquire();
    AVFrame *out_frame_p = out_frame.get();
    CHECK(graph->Pop(&out_frame_p)) << "Error fetch filtered frame.";

    auto tmp = AsNDArray(out_frame);
    // ── Backpressure: if the frame queue is full, wait for consumer ──
    // Prevents unbounded queue growth when Python consumes frames slower
    // than the decoder produces them.  0 (max_queue_frames_ default) or
    // negative disables backpressure entirely.
    if (max_queue_frames_ > 0) {
        std::unique_lock<std::mutex> blk(bp_mutex_);
        bp_cv_.wait(blk, [&] {
            return !run_.load()
                   || frame_queue_->Size() < static_cast<size_t>(max_queue_frames_);
        });
    }
    if (!run_.load()) return;
    pf_f_filter.stop();
    pf_f_push.start();
    if (out_buf.defined()) {
        CHECK(out_buf.Size() == tmp.Size());
        out_buf.CopyFrom(tmp);
        frame_queue_->Push(out_buf);
        ++frame_count_;
    } else {
        frame_queue_->Push(tmp);
        ++frame_count_;
    }
    if (on_output_) on_output_();
    pf_f_push.stop();
    if (DECORD_PROFILE && pf_f_filter.n % 3000 == 2999) {
        std::cerr << "[P2] d_send=" << pf_d_send.acc / pf_d_send.n
            << " d_recv=" << pf_d_recv.acc / pf_d_recv.n
            << " d_push=" << pf_d_push.acc / pf_d_push.n
            << " f_pop=" << pf_f_pop.acc / pf_f_pop.n
            << " f_filter=" << pf_f_filter.acc / pf_f_filter.n
            << " f_push=" << pf_f_push.acc / pf_f_push.n << std::endl;
    }
}

// Decode-side enqueue: discard-pts check happens here (the filter thread
// must not touch discard_pts_); the frame object ownership moves to the
// raw queue, so every decoded frame needs its own AVFramePtr.
void FFMPEGThreadedDecoder::EnqueueRawFrame(AVFramePtr frame) {
    frame->pts = frame->best_effort_timestamp;
    bool skip = false;
    {
      std::lock_guard<std::mutex> lock(pts_mutex_);
      skip = discard_pts_.find(frame->pts) != discard_pts_.end();
    }
    RawItem item;
    item.frame = frame;
    item.pts = frame->pts;
    item.kind = skip ? RawKind::Skip : RawKind::Frame;
    item.seq = raw_seq_++;
    // Backpressure: the decode thread must not run ahead of the filter /
    // consumer.  raw_queue_ and frame_queue_ are the decoded-frame buffers;
    // without a bound here they grow without limit when the consumer is
    // slower than the decoder (worst under CPU contention with two readers).
    // Blocking on the in-flight count propagates the consumer's pace all the
    // way back to Push(), bounding total memory.  Applied for every kind so
    // even the tiny markers can't starve the pipeline at the tail.
    if (max_queue_frames_ > 0) {
        // 条件变量背压（替代 1ms 睡眠轮询）：睡眠量子会把解码节流到
        // ~1ms/帧上限（rgb 实测 803fps 封顶）。Pop 侧唤醒，精确随消费
        // 步进；内存上界不变。
        size_t cap = static_cast<size_t>(max_queue_frames_ + DECORD_RAW_SLACK_FRAMES);
        std::unique_lock<std::mutex> lk(bp_mutex_);
        bp_cv_.wait(lk, [&] {
            return !run_.load()
                   || raw_queue_->Size() + frame_queue_->Size() < cap;
        });
        if (!run_.load()) return;
    }
    raw_queue_->Push(item);
}

void FFMPEGThreadedDecoder::FilterWorkerThread() {
    try {
        FilterWorkerThreadImpl();
    } catch (dmlc::Error error) {
        RecordInternalError(error.what());
        run_.store(false);
        frame_queue_->SignalForKill(); // Unblock all consumers
    }
}

void FFMPEGThreadedDecoder::FilterWorkerThreadImpl() {
    while (run_.load()) {
        RawItem item;
        pf_f_pop.start();
        if (!raw_queue_->Pop(&item)) {
            return;
        }
        pf_f_pop.stop();
        switch (item.kind) {
        case RawKind::Skip: {
            // keep the historical empty marker so NextFrameImpl's retry
            // loop skips the frame
            NDArray empty = NDArray::Empty({1}, kUInt8, kCPU);
            empty.pts = item.pts;
            frame_queue_->Push(empty);
            ++frame_count_;
            if (on_output_) on_output_();
            break;
        }
        case RawKind::Eof: {
            // mid-stream EOF marker (historical behaviour)
            frame_queue_->Push(NDArray());
            ++frame_count_;
            break;
        }
        case RawKind::DrainEnd: {
            // EOF drain finished on the decode side: emit the drain
            // markers the consumer recognises (kInt64 size-1 arrays)
            for (int cnt = 0; cnt < ThreadedDecoderInterface::kDrainMarkerCount; ++cnt) {
                frame_queue_->Push(NDArray::Empty({1}, kInt64, kCPU));
                ++frame_count_;
            }
            if (on_output_) on_output_();
            draining_.store(false);
            break;
        }
        case RawKind::Frame: {
            NDArray out_buf;
            bool get_buf = buffer_queue_->Pop(&out_buf);
            if (!get_buf) return;
            ProcessFrame(item.frame, out_buf);
            break;
        }
        }
    }
}

void FFMPEGThreadedDecoder::ConvertWorkerLoop() {
    // 线程本地 filter graph：各 worker 独立 sws 实例（扇出并行的前提）。
    // graph_gen_ 变化（SetRoi/SetCodecContext 热切换）时按快照描述串重建。
    std::shared_ptr<FFMPEGFilterGraph> local_graph;
    int local_gen = -1;
    while (run_.load()) {
        RawItem item;
        if (!raw_queue_->Pop(&item)) return;
        std::vector<NDArray> outs;
        bool call_on_output = false;
        bool clear_draining = false;
        switch (item.kind) {
        case RawKind::Skip: {
            // 保留历史空 marker（NextFrameImpl 的重试逻辑依赖）
            NDArray empty = NDArray::Empty({1}, kUInt8, kCPU);
            empty.pts = item.pts;
            outs.push_back(empty);
            call_on_output = true;
            break;
        }
        case RawKind::Eof: {
            outs.push_back(NDArray());
            break;
        }
        case RawKind::DrainEnd: {
            for (int cnt = 0; cnt < ThreadedDecoderInterface::kDrainMarkerCount; ++cnt)
                outs.push_back(NDArray::Empty({1}, kInt64, kCPU));
            call_on_output = true;
            clear_draining = true;
            break;
        }
        case RawKind::Frame: {
            {
                std::lock_guard<std::mutex> lk(filter_mutex_);
                if (local_gen != graph_gen_.load() && dec_ctx_) {
                    // 同 BuildFilterGraph：先恢复流时基（Clear/flush 会把
                    // dec_ctx->time_base 冲成 0/1，buffersrc 拒绝）
                    dec_ctx_->time_base = time_base_;
                    local_graph.reset(new FFMPEGFilterGraph(
                        graph_descr_, dec_ctx_.get(), output_format_));
                    local_gen = graph_gen_.load();
                }
            }
            if (!local_graph) continue;
            pf_f_filter.start();
            local_graph->Push(item.frame.get());
            AVFramePtr out_frame = AVFramePool::Get()->Acquire();
            AVFrame *out_frame_p = out_frame.get();
            CHECK(local_graph->Pop(&out_frame_p)) << "Error fetch filtered frame.";
            NDArray tmp = AsNDArray(out_frame);
            if (max_queue_frames_ > 0) {
                while (frame_queue_->Size() >= static_cast<size_t>(max_queue_frames_)
                       && run_.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
            if (!run_.load()) return;
            pf_f_filter.stop();
            outs.push_back(tmp);
            call_on_output = true;
            break;
        }
        }
        EmitOrdered(item.seq, std::move(outs), call_on_output, clear_draining);
    }
}

void FFMPEGThreadedDecoder::EmitOrdered(uint64_t seq, std::vector<NDArray> &&outs,
                                        bool call_on_output, bool clear_draining) {
    // 重排发射：持锁完成"插入完成项 + 连续段出队 + 背压等待 + 入队"，
    // 保证 frame_queue_ 严格按解码序（锁内背压睡眠安全 —— 消费者不取
    // 此锁，不会有其他 worker 越序插入）。
    std::lock_guard<std::mutex> lk(reorder_mu_);
    reorder_map_.emplace(seq, std::move(outs));
    while (reorder_map_.count(reorder_next_)) {
        auto it = reorder_map_.find(reorder_next_);
        std::vector<NDArray> batch = std::move(it->second);
        reorder_map_.erase(it);
        ++reorder_next_;
        for (auto &a : batch) {
            if (max_queue_frames_ > 0) {
                std::unique_lock<std::mutex> blk(bp_mutex_);
                bp_cv_.wait(blk, [&] {
                    return !run_.load()
                           || frame_queue_->Size() < static_cast<size_t>(max_queue_frames_);
                });
            }
            if (!run_.load()) return;
            frame_queue_->Push(a);
            ++frame_count_;
        }
        if (call_on_output && on_output_) on_output_();
        if (clear_draining) draining_.store(false);
    }
}

void FFMPEGThreadedDecoder::WorkerThread() {
    try {
        WorkerThreadImpl();
    } catch (dmlc::Error error) {
        RecordInternalError(error.what());
        run_.store(false);
        frame_queue_->SignalForKill(); // Unblock all consumers
        raw_queue_->SignalForKill();
    }
}

void FFMPEGThreadedDecoder::WorkerThreadImpl() {
    while (run_.load()) {
        AVPacketPtr pkt;

        int got_picture;
        bool ret = pkt_queue_->Pop(&pkt);
        if (!ret) {
            return;
        }
        if (!pkt) {
            // ── draining mode: pull buffered frames out of avcodec ──
            CHECK_GE(avcodec_send_packet(dec_ctx_.get(), NULL), 0) << "Thread worker: Error entering draining mode.";
            while (true) {
                AVFramePtr frame = AVFramePool::Get()->Acquire();
                got_picture = ReceiveFrame(dec_ctx_.get(), frame.get());
                if (got_picture == AVERROR_EOF) {
                    // signal the filter thread to emit drain markers
                    for (int cnt = 0; cnt < ThreadedDecoderInterface::kDrainMarkerCount; ++cnt) {
                        raw_queue_->Push(RawItem{AVFramePtr(), RawKind::DrainEnd, 0});
                    }
                    break;
                }
                EnqueueRawFrame(frame);
            }
        } else if (codec_is_av1_) {
            // ── AV1（dav1d）批量模式 ──
            // dav1d 的帧并行需要多个 packet 同时在解码器中积累；通用
            // drain-then-send 节奏每次只留 ~1-2 帧在途 → dav1d 内部
            // 线程池只有 ~3 核在跑（实测 cpu/wall=3.0，16 核 AV1 软解
            // 仅 ~300fps，多核扩展被节奏扼杀）。批量 send：连续 send
            // 直到 EAGAIN（解码器在途队列满）或攒够 kAV1BatchMax，
            // 再批量 receive 到 EAGAIN（EAGAIN = 在途帧仍在并行解码）。
            // dav1d 无 B 帧参考帧依赖（与 h264 帧线程不同），批量 send
            // 安全；receive 用普通语义（SYNCHRONOUS 会阻塞到帧完成，
            // 同样扼杀帧并行）。
            AVPacketPtr pending = std::move(pkt);
            int batch_sent = 0;
            const int kAV1BatchMax = 16;
            while (run_.load()) {
                if (!pending) {
                    if (!pkt_queue_->Pop(&pending)) return;
                    if (!pending) {
                        // 哨兵 → 转 drain（与通用模式相同）
                        CHECK_GE(avcodec_send_packet(dec_ctx_.get(), NULL), 0)
                            << "Thread worker: Error entering draining mode.";
                        while (true) {
                            AVFramePtr frame = AVFramePool::Get()->Acquire();
                            got_picture = ReceiveFrame(dec_ctx_.get(), frame.get());
                            if (got_picture == AVERROR_EOF) {
                                for (int cnt = 0;
                                     cnt < ThreadedDecoderInterface::kDrainMarkerCount;
                                     ++cnt) {
                                    raw_queue_->Push(RawItem{
                                        AVFramePtr(), RawKind::DrainEnd, 0});
                                }
                                break;
                            }
                            EnqueueRawFrame(frame);
                        }
                        break;
                    }
                }
                int send_ret = avcodec_send_packet(dec_ctx_.get(), pending.get());
                if (send_ret == 0) {
                    pending.reset();
                    if (++batch_sent < kAV1BatchMax) {
                        continue;  // 继续积累在途帧（帧并行）
                    }
                } else if (send_ret == AVERROR(EAGAIN)) {
                    // 在途已满 → 批量 receive 腾空后重试 send
                } else {
                    LOG(FATAL) << "Thread worker: Error sending packet: "
                               << send_ret;
                }
                // 批量 receive：取走所有已完成的帧（EAGAIN = 在途仍在解）
                while (run_.load()) {
                    AVFramePtr f = AVFramePool::Get()->Acquire();
                    got_picture = avcodec_receive_frame(dec_ctx_.get(), f.get());
                    if (got_picture == 0) {
                        EnqueueRawFrame(f);
                        continue;
                    }
                    if (got_picture == AVERROR(EAGAIN)) {
                        break;  // 在途未完成，回到 send 继续积累
                    }
                    if (got_picture == AVERROR_EOF) {
                        raw_queue_->Push(
                            RawItem{AVFramePtr(), RawKind::Eof, 0});
                        break;
                    }
                    LOG(FATAL) << "Thread worker: Error decoding frame: "
                               << got_picture;
                }
                batch_sent = 0;
            }
        } else {
            // ── normal mode: drain-then-send rhythm ──
            // Receive every finished frame before the next send.  Frame
            // threading decodes several frames in parallel; receiving one
            // per loop made send hit EAGAIN (internal queue full) almost
            // every iteration and the 1ms EAGAIN sleep dominated (measured
            // d_send 0.72ms of a 0.71ms frame budget).  Draining first
            // keeps the frame-thread queue empty so send succeeds
            // immediately.
            //
            // NOTE: a fully batched "send N then receive N" loop was tried
            // and reverted — it produced h264 "reference picture missing"
            // warnings and pixel mismatches (frame-thread B-frame
            // dependency handling breaks when packets are accumulated
            // faster than frames are pulled).  The drain-first rhythm is
            // the fastest correct shape.
            int send_ret;
            pf_d_send.start();
            while ((send_ret = avcodec_send_packet(dec_ctx_.get(),
                                                   pkt.get())) == AVERROR(EAGAIN)) {
                // Drain every output frame that is ready before sleeping:
                // with frame threading, several frames can be pending, and
                // sleeping per frame would cap throughput at ~1 frame/ms.
                while (true) {
                    AVFramePtr drain_frame = AVFramePool::Get()->Acquire();
                    got_picture = ReceiveFrame(dec_ctx_.get(), drain_frame.get());
                    if (got_picture == 0) {
                        EnqueueRawFrame(drain_frame);
                    } else if (got_picture == AVERROR(EAGAIN) ||
                               got_picture == AVERROR_EOF) {
                        break;
                    } else {
                        LOG(FATAL) << "Thread worker: Error decoding frame: " << got_picture;
                    }
                }
                // Input queue slots are freed by the internal workers, not by
                // receive, so yield once before retrying the send.
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(DECORD_EAGAIN_SLEEP_MS));
                if (!run_.load()) return;
            }
            CHECK_GE(send_ret, 0) << "Thread worker: Error sending packet: "
                                  << send_ret;
            pf_d_send.stop();
            // Drain every finished frame before the next send.
            pf_d_recv.start();
            while (run_.load()) {
                AVFramePtr f = AVFramePool::Get()->Acquire();
                got_picture = avcodec_receive_frame(dec_ctx_.get(), f.get());
                if (got_picture == 0) {
                    pf_d_recv.stop();
                    pf_d_push.start();
                    EnqueueRawFrame(f);
                    pf_d_push.stop();
                    pf_d_recv.start();
                    continue;
                }
                if (got_picture == AVERROR(EAGAIN)) {
                    // frame threads still decoding — fine, next loop drains
                } else if (got_picture == AVERROR_EOF) {
                    // Unexpected mid-stream EOF; keep the historical empty
                    // marker so the consumer's EOF path handles it.
                    raw_queue_->Push(RawItem{AVFramePtr(), RawKind::Eof, 0});
                } else {
                    LOG(FATAL) << "Thread worker: Error decoding frame: " << got_picture;
                }
                break;
            }
            pf_d_recv.stop();
        }
        // free raw memories allocated with ffmpeg
        // av_packet_unref(pkt);
    }
}

NDArray FFMPEGThreadedDecoder::CopyToNDArray(AVFramePtr p) {
    CHECK(p) << "Error: converting empty AVFrame to DLTensor";
    CHECK(AVPixelFormat(p->format) == AV_PIX_FMT_RGB24
          || AVPixelFormat(p->format) == AV_PIX_FMT_GRAY8
          || AVPixelFormat(p->format) == AV_PIX_FMT_YUV420P
          || AVPixelFormat(p->format) == AV_PIX_FMT_NV12)
        << "Only support RGB24/GRAY8/YUV420P/NV12 image to NDArray conversion, given: "
        << AVPixelFormat(p->format);
    DLDevice ctx;
    CHECK(!p->hw_frames_ctx) << "Not supported hw_frames_ctx";
    ctx = kCPU;
    auto device_api = runtime::DeviceAPI::Get(ctx);
    if (AVPixelFormat(p->format) == AV_PIX_FMT_NV12) {
        // packed 2D 输出（与 yuv420p 分支同布局）：Y 平面 + 交错 UV 平面
        // 各为连续行 —— 两次 memcpy（linesize==w 时整块），无逐字节交错。
        int h = p->height;
        int w = p->width;
        int rows = h + (h + 1) / 2;
        NDArray arr = NDArray::Empty({rows, w}, kUInt8, ctx);
        uint8_t *to_ptr = static_cast<uint8_t *>(arr.data_->dl_tensor.data);
        const uint8_t *y_src = p->data[0];
        int uv_h = h / 2;
        if (p->linesize[0] == w) {
            std::memcpy(to_ptr, y_src, static_cast<size_t>(h) * w);
        } else {
            for (int y = 0; y < h; ++y) {
                std::memcpy(to_ptr + static_cast<int64_t>(y) * w,
                            y_src + static_cast<int64_t>(y) * p->linesize[0],
                            static_cast<size_t>(w));
            }
        }
        const uint8_t *uv_src = p->data[1];
        uint8_t *uv_dst = to_ptr + static_cast<int64_t>(h) * w;
        if (p->linesize[1] == w && uv_h > 0) {
            std::memcpy(uv_dst, uv_src, static_cast<size_t>(uv_h) * w);
        } else {
            for (int y = 0; y < uv_h; ++y) {
                std::memcpy(uv_dst + static_cast<int64_t>(y) * w,
                            uv_src + static_cast<int64_t>(y) * p->linesize[1],
                            static_cast<size_t>(w));
            }
        }
        arr.pts = p->pts;
        return arr;
    }
    if (AVPixelFormat(p->format) == AV_PIX_FMT_YUV420P) {
        // packed 2D 输出：前 h 行原始 Y，随后 ceil(h/2) 行 interleaved
        // U/V（原始 4:2:0，按 MPEG-2 siting 打包）。Y 不做 range 展开：
        // 调用方通过 get_color_range() 自行展开（与 gray 输出同一语义）。
        int h = p->height;
        int w = p->width;
        int rows = h + (h + 1) / 2;
        NDArray arr = NDArray::Empty({rows, w}, kUInt8, ctx);
        uint8_t *to_ptr = static_cast<uint8_t *>(arr.data_->dl_tensor.data);
        // Y：原始值逐行拷贝（调用方按 get_color_range 自行展开；
        // gray 输出仍在此处展开，两者语义清晰分离）
        for (int y = 0; y < h; ++y) {
            const uint8_t *src = p->data[0] + static_cast<int64_t>(y) * p->linesize[0];
            uint8_t *dst = to_ptr + static_cast<int64_t>(y) * w;
            std::memcpy(dst, src, static_cast<size_t>(w));
        }
        // U/V：交错打包成 NV12 行（每对 luma 列一个 U、一个 V）
        int uv_w = (w + 1) / 2;
        int uv_h = h / 2;
        for (int y = 0; y < uv_h; ++y) {
            const uint8_t *u = p->data[1] + static_cast<int64_t>(y) * p->linesize[1];
            const uint8_t *v = p->data[2] + static_cast<int64_t>(y) * p->linesize[2];
            uint8_t *dst = to_ptr + static_cast<int64_t>(h + y) * w;
            for (int x = 0; x < w / 2; ++x) {
                int sx = std::min(x, uv_w - 1);
                dst[x * 2] = u[sx];
                dst[x * 2 + 1] = v[sx];
            }
        }
        arr.pts = p->pts;
        return arr;
    }
    int channel = AVPixelFormat(p->format) == AV_PIX_FMT_RGB24 ? 3 : 1;
    // CHECK(p->linesize[0] % p->width == 0)
    //     << "AVFrame data is not a compact array. linesize: " << p->linesize[0]
    //     << " width: " << p->width;

    NDArray arr = NDArray::Empty({p->height, p->width, channel}, kUInt8, ctx);
    void *to_ptr = arr.data_->dl_tensor.data;
    void *from_ptr = p->data[0];
    int linesize = p->width * channel;

    // arr.CopyFrom(&dlt);
    for (int i = 0; i < p->height; ++i) {
        // copy line by line
        device_api->CopyDataFromTo(
            from_ptr, i * p->linesize[0],
            to_ptr, i * linesize,
            linesize, ctx, ctx, kUInt8, nullptr);
    }
    arr.pts = p->pts;
    return arr;
}

static void AVFrameManagerDeleter(DLManagedTensor *manager) {
	delete static_cast<AVFrameManager*>(manager->manager_ctx);
	delete manager;
}

NDArray FFMPEGThreadedDecoder::AsNDArray(AVFramePtr p) {
    if (AVPixelFormat(p->format) == AV_PIX_FMT_YUV420P
            || AVPixelFormat(p->format) == AV_PIX_FMT_NV12) {
        // YUV420P 三平面 / NV12 双平面无法零拷贝单个 DLPack tensor ——
        // 打包成 NV12 布局的 2D 数组并拷贝（ROI-first 下只拷 ROI 尺寸）。
        // NV12 显式路由：双平面落进通用 compact-wrap 会缺 UV 行。
        return CopyToNDArray(p);
    }
    if (p->linesize[0] % p->width != 0) {
        // Fallback to copy since original AVFrame is not compact
        return CopyToNDArray(p);
    }
	DLManagedTensor* manager = new DLManagedTensor();
    auto av_manager = new AVFrameManager(p);
	manager->manager_ctx = av_manager;
	ToDLTensor(p, manager->dl_tensor, av_manager->shape);
	manager->deleter = AVFrameManagerDeleter;
	NDArray arr = NDArray::FromDLPack(manager);
    arr.pts = p->pts;
	return arr;
}

void FFMPEGThreadedDecoder::CheckErrorStatus() {
    if (error_status_.load()) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        LOG(FATAL) << error_message_;
    }
}

void FFMPEGThreadedDecoder::RecordInternalError(std::string message) {
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_ = message;
    }
    error_status_.store(true);
}

}  // namespace ffmpeg
}  // namespace decord
