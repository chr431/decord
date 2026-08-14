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
        virtual void SuggestDiscardPTS(std::vector<int64_t> dts) = 0;
        virtual void ClearDiscardPTS() = 0;
        virtual ~ThreadedDecoderInterface() = default;
};  // class ThreadedDecoderInterface

}  // namespace decord
#endif  // DECORD_VIDEO_THREADED_DECODER_INTERFACE_H_
