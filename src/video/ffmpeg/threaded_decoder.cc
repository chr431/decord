/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file threaded_decoder.cc
 * \brief FFmpeg threaded decoder Impl
 */

#include "threaded_decoder.h"

#include <dmlc/logging.h>
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
      max_queue_frames_(DECORD_CPU_FRAME_QUEUE_SIZE) {
}

void FFMPEGThreadedDecoder::SetCodecContext(AVCodecContext *dec_ctx, int width, int height, int rotation, int output_format) {
    bool running = run_.load();
    Clear();
    dec_ctx_.reset(dec_ctx);
    // LOG(INFO) << dec_ctx->width << " x " << dec_ctx->height << " : " << dec_ctx->time_base.num << " , " << dec_ctx->time_base.den;
    // std::string descr = "scale=320:240";
    // Force BT.601 (limited) color conversion: the CUDA decode path uses a
    // fixed BT.601 matrix (improc.cu), while FFmpeg's swscale honours the
    // stream's color metadata (most racing videos are tagged bt709).  The
    // two decoders then produce visibly different RGB for the same frame,
    // which breaks OCR on the CPU path (measured 28/30 recognition fails
    // and a systematic +7.5 G-channel offset).  setparams rewrites the
    // frame metadata so swscale picks the BT.601 matrix; the outputs then
    // match the GPU path within rounding (<=1-2 per pixel).
    char descr[160];
    const char *fmt = output_format ? "gray" : "rgb24";
    switch(rotation) {
        case 90:
            std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,transpose=1,scale=%d:%d,format=%s", width, height, fmt);
            break;
        case 180:
            std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,transpose=1,transpose=1,scale=%d:%d,format=%s", width, height, fmt);
            break;
        case 270:
            std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,transpose=2,scale=%d:%d,format=%s", width, height, fmt);
            break;
        case 0:
        default:
            std::snprintf(descr, sizeof(descr), "setparams=colorspace=bt470bg:color_primaries=bt470bg:color_trc=bt709,scale=%d:%d,format=%s", width, height, fmt);
    }
    filter_graph_ = FFMPEGFilterGraphPtr(new FFMPEGFilterGraph(descr, dec_ctx_.get(), output_format));
    if (running) {
        Start();
    }
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
    CHECK(run_.load());
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

FFMPEGThreadedDecoder::~FFMPEGThreadedDecoder() {
    Stop();
}

void FFMPEGThreadedDecoder::ProcessFrame(AVFramePtr frame, NDArray out_buf) {
    // filter image frame (format conversion, scaling...) — runs on the
    // filter worker thread, concurrent with the decode worker.
    pf_f_filter.start();
    filter_graph_->Push(frame.get());
    AVFramePtr out_frame = AVFramePool::Get()->Acquire();
    AVFrame *out_frame_p = out_frame.get();
    CHECK(filter_graph_->Pop(&out_frame_p)) << "Error fetch filtered frame.";

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
        if (!filter_graph_) return;
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
        // CHECK(filter_graph_) << "FilterGraph not initialized.";
        if (!filter_graph_) return;
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
    // int channel = p->linesize[0] / p->width;
    CHECK(AVPixelFormat(p->format) == AV_PIX_FMT_RGB24 || AVPixelFormat(p->format) == AV_PIX_FMT_GRAY8)
        << "Only support RGB24/GRAY8 image to NDArray conversion, given: "
        << AVPixelFormat(p->format);
    int channel = AVPixelFormat(p->format) == AV_PIX_FMT_RGB24 ? 3 : 1;
    // CHECK(p->linesize[0] % p->width == 0)
    //     << "AVFrame data is not a compact array. linesize: " << p->linesize[0]
    //     << " width: " << p->width;

    DLDevice ctx;
    CHECK(!p->hw_frames_ctx) << "Not supported hw_frames_ctx";
    ctx = kCPU;
    NDArray arr = NDArray::Empty({p->height, p->width, channel}, kUInt8, ctx);
    auto device_api = runtime::DeviceAPI::Get(ctx);
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
