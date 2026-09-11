/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file threaded_decoder.h
 * \brief FFmpeg threaded decoder definition
 */

#ifndef DECORD_VIDEO_FFMPEG_THREADED_DECODER_H_
#define DECORD_VIDEO_FFMPEG_THREADED_DECODER_H_

#include "filter_graph.h"
#include "../threaded_decoder_interface.h"
#include <decord/runtime/ndarray.h>

#include <chrono>
#include <functional>
#include <thread>
#include <unordered_set>
#include <mutex>
#include <map>
#include <atomic>
#include <vector>

#include <dmlc/concurrency.h>

namespace decord {
namespace ffmpeg {

class FFMPEGThreadedDecoder final : public ThreadedDecoderInterface {
    using PacketQueue = dmlc::ConcurrentBlockingQueue<AVPacketPtr>;
    using PacketQueuePtr = std::unique_ptr<PacketQueue>;
    using FrameQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using FrameQueuePtr = std::unique_ptr<FrameQueue>;
    using BufferQueue = dmlc::ConcurrentBlockingQueue<NDArray>;
    using BufferQueuePtr = std::unique_ptr<BufferQueue>;
    using FFMPEGFilterGraphPtr = std::shared_ptr<FFMPEGFilterGraph>;
    /*! \brief kind of an item on the raw (pre-filter) frame queue. */
    enum class RawKind { Frame, Skip, DrainEnd, Eof };
    struct RawItem {
        AVFramePtr frame;
        RawKind kind;
        int64_t pts;
        uint64_t seq;   // 解码序（转换扇出池乱序完成 → 按序发射）
        // explicit ctors: NSDMI + brace-init would make this a non-aggregate
        // under C++11 (aggregate NSDMI needs C++14); MSVC accepts it as an
        // extension but GCC/clang reject RawItem{...} — keep C++11 portable.
        RawItem() : frame(nullptr), kind(RawKind::Frame), pts(0), seq(0) {}
        RawItem(AVFramePtr f, RawKind k, int64_t p)
            : frame(std::move(f)), kind(k), pts(p), seq(0) {}
    };
    using RawFrameQueue = dmlc::ConcurrentBlockingQueue<RawItem>;
    using RawFrameQueuePtr = std::unique_ptr<RawFrameQueue>;

    public:
        FFMPEGThreadedDecoder();
        void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1,
                             int rotation = 0, int output_format = 0);
        void SetRoi(int x1, int y1, int x2, int y2) override;
        void Start();
        void Stop();
        void Clear();
        void Push(ffmpeg::AVPacketPtr pkt, runtime::NDArray buf);
        bool Pop(runtime::NDArray *frame);
        bool Drained() const override;
        /*! 产出回调（filter 线程调用）：混合解码器用它即时唤醒上载
         *  线程（替代 1ms 轮询）。 */
        void SetOnOutput(std::function<void()> cb) { on_output_ = std::move(cb); }
        void SuggestDiscardPTS(std::vector<int64_t> dts);
        void ClearDiscardPTS();
        ~FFMPEGThreadedDecoder();
        /*! 生产侧解码速率（帧/秒）。**唯一口径 = 滑窗持续产能**：最近 8 折
         *  （16 帧/折）内"连续 4 折最小值"的最大值 + 0.997/折慢降锁，≥4 折
         *  才出版（此前 0 = 未学得，调度走 CPU-sample chunk 冷启动）。
         *  2026-09-12 起替代容量跟踪 EWMA（快升 1.3x/慢降 5%）——后者
         *  实测缺陷（DECORD_CPU_RATE_DEBUG 折率轨迹，hevc hybrid 3000
         *  帧）：filter 每次从背压封堵脱封，头 1-3 折读到的是**解码器内部
         *  预跑存货的排空速度**（r=2491..9138 孤峰），EWMA 被逐峰棘轮永久
         *  卡 1200-2500，真值 ≈ 纯 CPU 单臂 754 → BuildPlan 冻结高估 rc →
         *  hevc CPU 份额翻 2-3x → 混跑 1568-1900 < 纯 nvdec 2036。
         *  sustained 排除孤峰（撑不满 4 折）又保 h264 突发产能（连续 20+
         *  折；纯中位数会在饿供相位把 h264 误杀到 rc=995 → 混跑 1881，
         *  实测否决）。
         *
         *  两路统一采用（cpu-out/gpu-out）。曾观测 gpu-out+sustained 的
         *  hevc 785fps 崩盘 / av1 挂死，归因为倾斜计划下的队头尾帧死锁
         *  （open-GOP DPB 扣 IDR+leading 等后续包）——已由 kick 突发治本
         *  （HybridThreadedDecoder::KICK_BURST，24/24 慢消费者压测 +
         *  金标 28/28）。转默认实测（引擎 GPU 管线全片 e2e，配对 3 遍）：
         *  hevc 12.73→9.66s（**−24%**，3/3）、av1 持平（±1% 漂移内）、
         *  h264 3.30→3.15s（−4.7%，3/3）；decode-only 3000f hevc cpu-out
         *  2684-2831 vs EWMA 1951（+43%，达并行理想 2790 的 96-101%）。 */
        double ProductionRate() const {
            return prod_rate_.load(std::memory_order_relaxed);
        }
        /*! \brief 已完成的折数（滑窗冷启动门控用） */
        int64_t ProductionFolds() const {
            return prod_folds_.load(std::memory_order_relaxed);
        }
        /*! 混合解码器用：放大 frame_queue_ 背压深度。默认 32 帧 —— 混合
         *  管线里 CPU chunk 的发射靠 cpu_ready_ 存货瞬时完成，存货攒不到
         *  一个 chunk 帧数（~286）就会退化为实时跟随解码速率
         *  （hevc CPU 侧 ~700fps，实测整体被拖到 0.70x）。 */
        void SetQueueDepth(int n) { max_queue_frames_ = n; }
        /*! \brief 可发射存货深度（filter 后帧队列）。混合调度的库存迟滞
         *  切换信号：GPU 块把它灌满、CPU 块把它排空 —— 份额自发涌现。 */
        size_t QueueDepth() const {
            return frame_queue_ ? frame_queue_->Size() : 0;
        }
        /*! \brief 在途深度（未解码包 + 已解码未 filter 的原始帧）。
         *  与 QueueDepth 的差值用于诊断"包在途却不产出"的停摆层。 */
        size_t PendingDepth() const {
            return (pkt_queue_ ? pkt_queue_->Size() : 0)
                 + (raw_queue_ ? raw_queue_->Size() : 0);
        }
    private:
        void WorkerThread();
        void WorkerThreadImpl();
        void FilterWorkerThread();
        void FilterWorkerThreadImpl();
        // ── RGB 转换扇出池（convert_workers_ > 1 时启用）──
        // 帧级并行：解码单线程，转换分发到 N 个 worker（各持独立
        // filter graph/sws 实例），完成项按解码序重排后入 frame_queue_。
        // 仅 RGB 输出启用（转换占主导）；yuv420/gray 走单线程旧路径，
        // hybrid（yuv420/gray）行为零变化。
        void StartConvertPool();
        void StopConvertPool();
        void ConvertWorkerLoop();
        void EmitOrdered(uint64_t seq, std::vector<NDArray> &&outs,
                         bool call_on_output, bool clear_draining);
        std::vector<std::thread> convert_worker_threads_;
        std::mutex reorder_mu_;
        std::mutex bp_mutex_;           // 背压 cv（解码/转换线程 ↔ 消费者 Pop）
        std::condition_variable bp_cv_;
        std::map<uint64_t, std::vector<NDArray> > reorder_map_;
        uint64_t reorder_next_ = 0;
        uint64_t raw_seq_ = 0;      // 仅解码线程推进
        int convert_workers_ = 1;
        std::string graph_descr_;   // BuildFilterGraph 描述串快照
        std::atomic<int> graph_gen_{0};
        void EnqueueRawFrame(AVFramePtr frame);
        void RecordInternalError(std::string message);
        void CheckErrorStatus();
        void ProcessFrame(AVFramePtr p, NDArray out_buf);
        NDArray CopyToNDArray(AVFramePtr p);
        NDArray AsNDArray(AVFramePtr p);
        /*! \brief 依当前 roi_/旋转/缩放/格式参数重建 filter 图。 */
        void BuildFilterGraph();
        PacketQueuePtr pkt_queue_;
        /*! \brief decoded (pre-filter) frames, consumed by the filter thread.
         *  Two-stage pipeline: the decode thread only drives avcodec (frame
         *  threads stay busy), the filter thread runs sws_scale conversion
         *  concurrently — the old single worker serialised them and the
         *  filter cost (~0.54ms @1080p) starved the frame threads. */
        RawFrameQueuePtr raw_queue_;
        FrameQueuePtr frame_queue_;
        BufferQueuePtr buffer_queue_;
        std::atomic<int> frame_count_;
        std::atomic<bool> draining_;
        std::thread t_;          // decode worker
        std::thread filter_t_;   // filter worker
        std::atomic<bool> run_;
        FFMPEGFilterGraphPtr filter_graph_;
        std::mutex filter_mutex_;   // 保护 filter_graph_ 热切换（SetRoi）
        AVCodecContextPtr dec_ctx_;
        std::unordered_set<int64_t> discard_pts_;
        std::mutex pts_mutex_;
        std::mutex error_mutex_;
        std::atomic<bool> error_status_;
        std::string error_message_;
        int max_queue_frames_;
        std::function<void()> on_output_;
        // ── 生产侧速率（仅 filter 线程访问，除原子速率外）──
        std::atomic<double> prod_rate_{0.0};   ///< 滑窗持续产能读数（唯一口径）
        std::chrono::steady_clock::time_point last_prod_tp_{};
        int64_t prod_seg_frames_ = 0;
        double prod_seg_secs_ = 0.0;
        std::atomic<int64_t> prod_folds_{0};   ///< 已完成折数（计划冷启动门控）
        static constexpr int kProdFoldRing = 8;  ///< 滑窗折数（8×16=128 帧）
        double prod_fold_ring_[kProdFoldRing] = {0};
        int prod_fold_i_ = 0;   ///< 环形写位（仅 filter 线程）
        int prod_fold_n_ = 0;   ///< 有效样本数（封顶 kProdFoldRing）
        // AV1（dav1d）解码：批量 send 模式（dav1d 帧并行需多 packet 在途）
        bool codec_is_av1_ = false;
        // ── ROI-first 状态（SetRoi）──
        int roi_x1_ = 0, roi_y1_ = 0, roi_x2_ = -1, roi_y2_ = -1;  // 半开
        bool roi_valid_ = false;
        int orig_w_ = 0, orig_h_ = 0;   // 解码原始分辨率（未旋转）
        int out_w_ = -1, out_h_ = -1;   // SetCodecContext 传入的目标尺寸
        int rotation_ = 0;
        int output_format_ = 0;
        int color_range_ = 0;             // 0 = limited/tv, 1 = full/pc（Y 展开）
        AVRational time_base_{0, 1};    // 流 time_base 快照（flush 会重置）

    DISALLOW_COPY_AND_ASSIGN(FFMPEGThreadedDecoder);
};

}  // namespace ffmpeg
}  // namespace decord

#endif  // DECORD_VIDEO_FFMPEG_THREADED_DECODER_H_
