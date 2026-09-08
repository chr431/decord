/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file video_decoder_interface.h
 * \brief Video Decoder Interface
 */

#ifndef DECORD_VIDEO_THREADED_DECODER_INTERFACE_H_
#define DECORD_VIDEO_THREADED_DECODER_INTERFACE_H_

#include "ffmpeg/ffmpeg_common.h"
#include <vector>
#include <decord/runtime/ndarray.h>

namespace decord {
typedef enum {
    DECORD_SKIP_FRAME   = 0x01,   /**< Set when the frame is not wanted, we can skip image processing  */
} ThreadedDecoderFlags;

class ThreadedDecoderInterface {
    public:
        // Number of output buffers the hardware decoder keeps in flight.
        // PushNext uses this to size the EOF drain so display callbacks
        // can always pop a buffer instead of blocking.
        static constexpr int kMaxOutputSurfaces = 20;
        // Number of kInt64 "draining finished" markers pushed after EOF;
        // NextFrameImpl interprets them as end-of-stream signals and falls
        // back to cached frames / rewind recovery.
        static constexpr int kDrainMarkerCount = 128;

        virtual void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1, int rotation = 0, int output_format = 0) = 0;
        /*!
         * \brief 固定 ROI 输出（ROI-first 解码管线，须在任何帧解码前调用）。
         * \param x1,y1,x2,y2 半开区间 [x1,x2) x [y1,y2)（全帧坐标）。
         *        w<=0 或 h<=0 表示清除 ROI（回退全帧输出 + 调用方裁剪）。
         * 语义：解码器从此只输出该矩形（CPU: filter 图先 crop 再格式转换；
         * GPU: 转换 kernel 只处理 ROI 窗口，池缓冲缩小为 ROI 尺寸）。
         */
        virtual void SetRoi(int x1, int y1, int x2, int y2) = 0;
        virtual void Start() = 0;
        virtual void Stop() = 0;
        virtual void Clear() = 0;
        virtual void Push(ffmpeg::AVPacketPtr pkt, runtime::NDArray buf) = 0;
        virtual bool Pop(runtime::NDArray *frame) = 0;
        /*!
         * \brief Whether every decoded output (including EOF drain markers)
         *        has been consumed.  Used to distinguish "no frame right now"
         *        from "EOF and the decoder can never produce another frame",
         *        which matters for the non-blocking GPU Pop().
         */
        virtual bool Drained() const = 0;
        /*! 混合解码器建议的 demux 领先深度（包数；0 = 不建议）。
         *  VideoReader 取 max(env 基线, 本值) —— 深度是离峰生产的前提，
         *  由解码器按其自适应预算给出。 */
        virtual int SuggestPrefetchDepth() const { return 0; }
        /*! 解码器是否真的需要更多 demux 包（默认恒真 = 旧行为）。
         *  NextFrameImpl 的重试推包用它门控：hybrid 侧丢弃的帧
         *  （stale-drop/kick 陈旧帧）永不计入 VideoReader 的
         *  frames_popped_，其 pkts_pushed_-frames_popped_ 在途账目
         *  只会单调虚高（不可用于限流，D3 教训）；而 hybrid 内部的
         *  side_pending_ 逐包递增、发射/丢弃逐帧核销，是精确在途。
         *  无条件重试推包曾以消费轮询速度把 demux 拉到解码前面
         *  10+ chunks —— 盲阶段路由决策全部跑在速率学习之前。 */
        virtual bool NeedsPackets() const { return true; }
        virtual void SuggestDiscardPTS(std::vector<int64_t> dts) = 0;
        virtual void ClearDiscardPTS() = 0;
        virtual ~ThreadedDecoderInterface() = default;
};  // class ThreadedDecoderInterface

}  // namespace decord
#endif  // DECORD_VIDEO_THREADED_DECODER_INTERFACE_H_
