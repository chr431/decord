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
    const char *fmt = output_format_ == 2 ? "yuv420p"
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
        auto ft = std::thread(&FFMPEGThreadedDecoder::FilterWorkerThread, this);
        std::swap(filter_t_, ft);
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
    }
    if (t_.joinable()) {
        // LOG(INFO) << "joining";
        t_.join();
    }
    if (filter_t_.joinable()) {
        filter_t_.join();
    }
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
    AVFramePtr out_frame = AVFramePool::Get()->Acquire();
    AVFrame *out_frame_p = out_frame.get();
    CHECK(graph->Pop(&out_frame_p)) << "Error fetch filtered frame.";

    auto tmp = AsNDArray(out_frame);
    // ── Backpressure: if the frame queue is full, wait for consumer ──
    // Prevents unbounded queue growth when Python consumes frames slower
    // than the decoder produces them.  0 (max_queue_frames_ default) or
    // negative disables backpressure entirely.
    if (max_queue_frames_ > 0) {
        while (frame_queue_->Size() >= static_cast<size_t>(max_queue_frames_)
               && run_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
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
    // Backpressure: the decode thread must not run ahead of the filter /
    // consumer.  raw_queue_ and frame_queue_ are the decoded-frame buffers;
    // without a bound here they grow without limit when the consumer is
    // slower than the decoder (worst under CPU contention with two readers).
    // Blocking on the in-flight count propagates the consumer's pace all the
    // way back to Push(), bounding total memory.  Applied for every kind so
    // even the tiny markers can't starve the pipeline at the tail.
    if (max_queue_frames_ > 0) {
        size_t cap = static_cast<size_t>(max_queue_frames_ + DECORD_RAW_SLACK_FRAMES);
        while (run_.load()
               && raw_queue_->Size() + frame_queue_->Size() >= cap) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
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
          || AVPixelFormat(p->format) == AV_PIX_FMT_YUV420P)
        << "Only support RGB24/GRAY8/YUV420P image to NDArray conversion, given: "
        << AVPixelFormat(p->format);
    DLDevice ctx;
    CHECK(!p->hw_frames_ctx) << "Not supported hw_frames_ctx";
    ctx = kCPU;
    auto device_api = runtime::DeviceAPI::Get(ctx);
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
    if (AVPixelFormat(p->format) == AV_PIX_FMT_YUV420P) {
        // YUV420P 有三个平面（Y/U/V），无法零拷贝单个 DLPack tensor ——
        // 打包成 NV12 布局的 2D 数组并拷贝（ROI-first 下只拷 ROI 尺寸）。
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
