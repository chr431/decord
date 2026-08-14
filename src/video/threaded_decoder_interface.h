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
        // 异步批解码（v0.7.9）：解码器 surface 总数（驱动上限 32，实测
        // 64 被 cuvidCreateDecoder 拒绝）。必须大于最大在途帧数：pkt 流水
        // 门（kMaxPipelinePackets=12）+ 预取 8 + 显示延迟 4 ≈ 24 < 32 ——
        // 保证映射延迟到消费者批量 SyncStream 期间，parser 永不把还在
        // 映射的 surface 复用于新解码（否则解码覆写映射数据 / 双映射
        // 报错 CUDA error 205）。
        static constexpr int kDecodeSurfaceCount = 32;
        // Push() 背压门：pkt_queue_ 超过此值则等待（解码线程领先上限）。
        // 与 kDecodeSurfaceCount 解耦：surface 数固定 32，门收紧到 12
        // 留出 surface 复用安全余量。
        static constexpr int kMaxPipelinePackets = 12;
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
         * \brief GPU 异步批解码：同步解码流一次并释放全部待处理映射
         * （异步转换 kernel 完成的批级栅栏）。CPU 解码器为 no-op。
         * Pop 出的帧在 SyncStream 之前由同步拷贝（默认流全设备同步）
         * 保证顺序正确；本方法把 N 次逐帧 sync 合并为一次。
         */
        virtual void SyncStream() {}
        virtual void SuggestDiscardPTS(std::vector<int64_t> dts) = 0;
        virtual void ClearDiscardPTS() = 0;
        virtual ~ThreadedDecoderInterface() = default;
};  // class ThreadedDecoderInterface

}  // namespace decord
#endif  // DECORD_VIDEO_THREADED_DECODER_INTERFACE_H_
