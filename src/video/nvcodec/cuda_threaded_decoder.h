/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_decoder.h
 * \brief NVCUVID based decoder
 */

#ifndef DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_
#define DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_

#include "cuda_stream.h"
#include "cuda_parser.h"
#include "cuda_context.h"
#include "cuda_decoder_impl.h"
#include "cuda_texture.h"
#include "cuda_mapped_frame.h"
#include "../ffmpeg/ffmpeg_common.h"
#include "../threaded_decoder_interface.h"

#include <condition_variable>
#include <functional>
#include <thread>
#include <mutex>
#include <memory>
#include <deque>
#include <vector>

#include <decord/runtime/ndarray.h>
#include <dmlc/concurrency.h>
#include <dlpack/dlpack.h>

namespace decord {
namespace cuda {

class CUThreadedDecoder final : public ThreadedDecoderInterface {
    using NDArray = runtime::NDArray;
    using AVPacketPtr = ffmpeg::AVPacketPtr;
    using AVCodecContextPtr = ffmpeg::AVCodecContextPtr;
    using AVBSFContextPtr = ffmpeg::AVBSFContextPtr;
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacketPtr>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using BufferQueue = dmlc::ConcurrentBlockingQueue<CUVIDPARSERDISPINFO*>;
    using BufferQueuePtr = std::unique_ptr<BufferQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;
    using PermitQueue = dmlc::ConcurrentBlockingQueue<int>;
    using PermitQueuePtr = std::shared_ptr<PermitQueue>;
    using ReorderQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using ReorderQueuePtr = std::unique_ptr<ReorderQueue>;
    using FrameOrderQueue = dmlc::ConcurrentBlockingQueue<int64_t>;
    using FrameOrderQueuePtr = std::unique_ptr<FrameOrderQueue>;

    public:
        CUThreadedDecoder(int device_id, AVCodecParameters *codecpar, const AVInputFormat *iformat);
        void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1, int rotation = 0, int output_format = 0);
        void SetRoi(int x1, int y1, int x2, int y2) override;
        bool Initialized() const;
        void Start();
        void Stop();
        void Clear();
        void Push(AVPacketPtr pkt, NDArray buf);
        bool Pop(NDArray *frame);
        bool Drained() const override;
        /*! 产出回调（display 线程调用）：混合解码器用它即时唤醒
         *  落地/喂包线程（替代 1ms 轮询，hevc ~1800fps 下轮询延迟
         *  直接封顶吞吐）。 */
        void SetOnOutput(std::function<void()> cb) { on_output_ = std::move(cb); }
        /*! \brief stall 取证：CU 三条队列深度（包 / 输出缓冲 / 待落地重排
         *  环）。混合解码器 Pop 空手时用来自证 8 帧卡在哪一层：pkt>0 =
         *  解析线程停摆；pkt=0 而 bufs/ord 非零 = 解码/转换在途；
         *  三全零 = NVDEC DPB 扣留（等驱动它的后续包）。 */
        void DiagDepths(int64_t *pkt, int64_t *bufs, int64_t *ord) {
            *pkt = pkt_queue_ ? static_cast<int64_t>(pkt_queue_->Size()) : -1;
            *bufs = frame_queue_ ? static_cast<int64_t>(frame_queue_->Size()) : -1;
            *ord = reorder_queue_ ? static_cast<int64_t>(reorder_queue_->Size()) : -1;
        }
        void SuggestDiscardPTS(std::vector<int64_t> dts);
        void ClearDiscardPTS();
        ~CUThreadedDecoder();

        static int CUDAAPI HandlePictureSequence(void* user_data, CUVIDEOFORMAT* format);
        static int CUDAAPI HandlePictureDecode(void* user_data, CUVIDPICPARAMS* pic_params);
        static int CUDAAPI HandlePictureDisplay(void* user_data, CUVIDPARSERDISPINFO* disp_info);

    private:
        int HandlePictureSequence_(CUVIDEOFORMAT* format);
        int HandlePictureDecode_(CUVIDPICPARAMS* pic_params);
        int HandlePictureDisplay_(CUVIDPARSERDISPINFO* disp_info);
        void LaunchThread();
        void LaunchThreadImpl();
        void RecordInternalError(std::string message);
        void CheckErrorStatus();
        void FlushDeferred();
        void InitBitStreamFilter(AVCodecParameters *codecpar, const AVInputFormat *iformat);

        int device_id_;
        CUStream stream_;
        CUdevice device_;
        CUContext ctx_;
        CUVideoParser parser_;
        CUVideoDecoderImpl decoder_;
        PacketQueuePtr pkt_queue_;
        FrameQueuePtr frame_queue_;
        //BufferQueuePtr buffer_queue_;
        //std::unordered_map<int64_t, runtime::NDArray> reorder_buffer_;
        ReorderQueuePtr reorder_queue_;
        //FrameOrderQueuePtr frame_order_;
        // int64_t last_pts_;
        std::thread launcher_t_;
        //std::thread converter_t_;
        //std::vector<PermitQueuePtr> permits_;
        // std::vector<uint8_t> frame_in_use_;
        std::atomic<bool> run_;
        std::atomic<int> frame_count_;
        std::atomic<bool> draining_;

        CUTextureRegistry tex_registry_;
        AVRational nv_time_base_;
        AVRational frame_base_;
        AVCodecContextPtr dec_ctx_;
        // ── ROI-first 状态（SetRoi，须在 Start 前调用）──
        int roi_x1_ = 0, roi_y1_ = 0, roi_x2_ = -1, roi_y2_ = -1;  // 半开
        bool roi_valid_ = false;
        int orig_w_ = -1, orig_h_ = -1;  // 解码原始分辨率
        int out_w_ = -1, out_h_ = -1;    // SetCodecContext 目标尺寸
        int output_format_ = 0;          // 0 = RGB24, 1 = GRAY8, 2 = YUV420/NV12
        int color_range_ = 0;            // 0 = limited/tv, 1 = full/pc（Y 展开语义）
        /*! \brief AV bitstream filter context */
        AVBSFContextPtr bsf_ctx_;
        unsigned int width_;
        unsigned int height_;
        // uint64_t decoded_cnt_;
        std::unordered_set<int64_t> discard_pts_;
        std::mutex pts_mutex_;
        std::mutex error_mutex_;
        std::atomic<bool> error_status_;
        std::string error_message_;
        // packet-queue backpressure: Push() waits on this cv instead of
        // busy-polling with a 1ns sleep when the queue exceeds the limit
        // Deferred per-frame sync+unmap: the display callback used to run
        // map -> convert kernel -> cudaStreamSynchronize -> unmap per frame,
        // stalling the parser thread (sync callbacks inside
        // cuvidParseVideoData) on each GPU round trip while NVDEC idled.
        // Now: callback k starts with FlushDeferred() (sync+unmap frame k-1,
        // whose kernel finished during the last decode interval), and the
        // tail sync of frame k moves to the consumer-side Pop (first pop
        // syncs once, later pops are free). At most 2 frames stay mapped;
        // all CUDA calls remain on the parser thread.
        std::unique_ptr<CUMappedFrame> deferred_frame_;
        std::atomic<bool> deferred_valid_{false};
        /*! \brief 每帧转换完成事件环：display 回调在转换 kernel 入队后
         *  record，Pop 按出队序等待"自己这一帧"的事件 —— 替代旧的全流
         *  cudaStreamSynchronize（它会连尚未消费的后续帧转换一起等，
         *  消费者被生产超前深度串行化）。事件序 = reorder 出队序，与
         *  reorder_queue_ 严格 1:1（drain marker 对应空事件）。 */
        std::mutex ev_mtx_;
        std::deque<void *> frame_events_;   // cudaEvent_t，nullptr = 无需等待
        std::vector<void *> ev_ring_;       // 预建事件池（Start 创建，Stop 销毁）
        size_t ev_ring_next_ = 0;           // 仅 parser 线程推进
        /*! \brief 事件环管理 + 每帧事件 record/wait（见成员注释） */
        void CreateEventRing();
        void DestroyEventRing();
        void RecordFrameEvent();
        cudaEvent_t PopFrameEvent();
        std::mutex pkt_room_mutex_;
        std::function<void()> on_output_;
        std::condition_variable pkt_room_cv_;

    DISALLOW_COPY_AND_ASSIGN(CUThreadedDecoder);
};
}  // namespace cuda
}  // namespace decord
#endif  // DECORD_VIDEO_NVCODEC_CUDA_THREADED_DECODER_H_
