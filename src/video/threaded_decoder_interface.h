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

/*! \brief kInt64 drain marker 的构造与取值（窗口架构重做，2026-10-08）。
 *
 * 此前 marker 是 NDArray::Empty({1}, kInt64) 的**未初始化**标量——
 * 消费方只能凭 dtype 识别，窗缘与真 EOF 在协议上不可分（C-57 一族
 * 缺陷的构造性根源之一）。现在所有创建点写入判别值：
 *   0 = EOF 排空（子解码器一律铸 0；hybrid 转发层在真 EOF 时也铸 0）
 *   1 = 硬窗缘（仅 hybrid 的 Pop 转发层铸造——子解码器排空是侧语义，
 *       全局语义由合并层宣告） */
inline runtime::NDArray MakeDrainMarker(int64_t value) {
    runtime::NDArray m = runtime::NDArray::Empty({1}, kInt64, kCPU);
    // operator->() 只回 const DLTensor*，但 DLTensor::data 本就是
    // void*（可写）——经它落载荷。
    *static_cast<int64_t *>(m.operator->()->data) = value;
    return m;
}
/*! \brief 读 marker 载荷；非 marker/未初始化一律按 0（EOF 兼容旧读法）*/
inline int64_t DrainMarkerValue(const runtime::NDArray &m) {
    if (!m.defined()) return 0;
    const DLTensor *t = m.operator->();
    if (!t || t->dtype.code != kDLInt || t->dtype.bits != 64) return 0;
    if (!t->data || t->ndim != 1 || t->shape[0] != 1) return 0;
    return *static_cast<const int64_t *>(t->data);
}
/*! drain marker 判别值：真 EOF（子解码器与 hybrid 真 EOF 转发）*/
constexpr int64_t kDrainMarkerEOF = 0;
/*! drain marker 判别值：硬窗缘（仅 hybrid 合并层铸造）*/
constexpr int64_t kDrainMarkerWindowEnd = 1;

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
        /*! rief 终态取证（默认空）：VideoReader 在 EOF 重试 FATAL 前
         *  调用——死亡位点自拍，不可能被时序扰动避开（2026-09-28 停滞轮）*/
        virtual void DumpState(const char *tag) const { (void)tag; }
        /*! rief 解码臂冻结标志（2026-09-28 av1 死锁轮）：hybrid 的 CU
         *  臂启动竞态冻结时置位；VideoReader 据此自愈（Seek 重解）。 */
        virtual bool ArmStalled() const { return false; }
        /*! rief 清除冻结标志（自愈动作完成后调用） */
        virtual void ClearArmStall() {}
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
        /*! hybrid 遥测快照（"k=v;k=v" 协议字符串；非 hybrid 实现返回空）。
         *  引擎层经 VideoReader.hybrid_stats() 取走并入 RunReport——一次
         *  extract 拿全部 fork 侧数据，不再依赖 stderr 文本/环境变量
         *  （2026-09-17 引擎穿透轮）。原子计数器快照，任意时刻可调。 */
        virtual std::string HybridStatsProbe() { return ""; }
        /*! 硬窗界（区间版，2026-10-08 重做）：声明消费区间的绝对帧号
         *  [lo, hi)。hybrid 实现据此硬性停止 demux 与 GOP 派工——窗口外
         *  一个包都不读（窗口内帧解码必需的边界 GOP 整体供给属例外）。
         *  由 VideoReader 在设窗与每次 seek 落锚时重推。默认空操作
         *  （cpu/nvdec 基础路径 prefetch 深度 8 包 + NeedsPackets 已
         *  如实化，无越窗问题）。 */
        virtual void SetDecodeWindowRange(int64_t lo, int64_t hi) {
            (void)lo; (void)hi;
        }
        /*! 旧计数窗接口（0.8.5 前语义）：仅 VideoReader 兼容保留，
         *  hybrid 已不覆写（区间版取代）；纯解码器本就是空操作。 */
        virtual void SetDecodeWindow(int64_t max_frames) { (void)max_frames; }
        virtual ~ThreadedDecoderInterface() = default;
};  // class ThreadedDecoderInterface

}  // namespace decord
#endif  // DECORD_VIDEO_THREADED_DECODER_INTERFACE_H_
