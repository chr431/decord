/*!
 *  Copyright (c) 2026 decord fork contributors
 * \file hybrid_threaded_decoder.h
 * \brief CPU+GPU 混合线程解码器：单 demux 流按关键帧 chunk 路由到
 *        FFMPEGThreadedDecoder(软解) 与 CUThreadedDecoder(NVDEC)，
 *        输出按 chunk 顺序合并 —— 对 VideoReader 透明地实现
 *        ThreadedDecoderInterface（seek / prefetch / skip / discard 全兼容）。
 *
 * GPU 数据通路（v2，有界流水线，demux 永不阻塞）：
 *
 *   demux 线程 Push ─► CPU 包 ─► FFMPEGThreadedDecoder（自带 36 帧背压）
 *                 └─► GPU 包 ─► gpu_pkt_q_（宿主 RAM，压缩包很小）
 *                                    │
 *   GPU 工作线程（GpuWorkerLoop）     ▼
 *     喂包: ready_ 有余量且池有空块时才从 gpu_pkt_q_ 取包推给
 *           CUThreadedDecoder（池耗尽即 GPU 解码超前已满，天然背压，
 *           越界的包留在宿主队列）；
 *     落地: 持续 gpu_->Pop（排空 NVDEC 输出 reorder 队列，防止显存
 *           无界堆积）并同步 D2H 成宿主 NDArray，排入 ready_
 *           （字节预算有界）。
 *   消费者 Pop 只从 ready_ / CPU 子解码器取宿主帧 —— D2H 不在消费
 *   关键路径上，显存占用 = 池缓冲(28 帧) + NVDEC 内部 surface，恒定。
 *
 * 死锁免疫论证：demux 线程的 Push（CPU 直推 / GPU 入宿主队列）与
 * 消费者线程的 Pop（非阻塞）都永不阻塞；GPU 工作线程内部全部非阻塞
 * 轮询（池 TryAcquire / CU Pop），唯一可能阻塞的是 CU Push 的包队列
 * 背压等待（由解析线程独立排空，自解）与同步 D2H（必然完成）。
 * ChooseSide 另有 GPU 待发射帧字节预算（kGpuAheadBytes）硬上限，
 * 防止调度失衡时压缩包队列无界增长。
 */

#ifndef DECORD_VIDEO_HYBRID_THREADED_DECODER_H_
#define DECORD_VIDEO_HYBRID_THREADED_DECODER_H_

#include "threaded_decoder_interface.h"
#include "ffmpeg/threaded_decoder.h"
#include "storage_pool.h"

#include <atomic>
#include <functional>
#include <functional>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

namespace decord {

#ifdef DECORD_USE_CUDA
/*!
 * \brief 有界 GPU 缓冲池（混合解码器专用）：固定 cap 块输出缓冲，
 *  Acquire 空时阻塞（等待落地线程回收），绝不超额分配 —— 显存占用的
 *  硬边界。Acquire 必须非阻塞：工作线程单线程串行 LandStep(回收池)→
 *  FeedStep(取池)，若取池阻塞则回收永不发生（互等死锁，实测 CPU→GPU
 *  切换即卡死）。回收经由 NDArray 自定义 deleter（仿 NDArrayPool，需其
 *  friendship），回收时唤醒等待者。
 */
class HybridGpuBufferPool {
  public:
    void Reset(std::size_t cap, std::vector<int64_t> shape,
               DLDataType dtype, DLDevice dev);
    ~HybridGpuBufferPool();
    void Start();
    void Stop();
    /*! \brief 尝试取一块缓冲；池空且已建满时返回 false（不阻塞） */
    bool Acquire(runtime::NDArray *out);
    /*! \brief 缓冲回收回调（Deleter 触发）：混合解码器用它即时唤醒
     *  喂包/上载线程（池耗尽时的重试延迟从 1ms 轮询降为即时）。 */
    void SetOnRelease(std::function<void()> cb) { on_release_ = std::move(cb); }
    /*! \brief 启用回池保序：Deleter 在 legacy 流 record 事件，Acquire
     *  后由调用方 cudaStreamWaitEvent —— 上载 H2D 无需 blocking 流全局
     *  互斥即可与消费侧在途拷贝保序（否则二者互相排队，上载被消费
     *  拷贝序列化，实测 hevc 上载链掉到 ~700fps）。 */
    void EnableReleaseSync();
    /*! \brief Acquire 拿到缓冲后调用：等待其上次消费拷贝完成 */
    void WaitRelease(void *stream);
    static void Deleter(runtime::NDArray::Container *ptr);

  private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<runtime::NDArray> free_;
    std::size_t cap_ = 0;
    std::size_t created_ = 0;
    bool running_ = false;
    std::vector<int64_t> shape_;
    DLDataType dtype_ = kUInt8;
    DLDevice dev_{kDLCUDA, 0};
    std::function<void()> on_release_;
    void *release_ev_ = nullptr;    // cudaEvent_t（回池点标记）
    void *sync_stream_ = nullptr;   // 等待事件的上载流
};
#endif  // DECORD_USE_CUDA

namespace cuda {
class CUThreadedDecoder;
}  // namespace cuda

class HybridThreadedDecoder : public ThreadedDecoderInterface {
  public:
    /*!
     * \brief 构造混合解码器（GPU 侧立即初始化 NVDEC 子解码器）。
     * \param device_id CUDA 设备号
     * \param codecpar 视频流参数（仅构造期使用：初始化 GPU 码流过滤器）
     * \param iformat 容器格式（仅构造期使用）
     * \note SetCodecContext 传入的 AVCodecContext 归 CPU 子解码器所有；
     *       GPU 子解码器的 AVCodecContext 在 SetCodecContext 内部从
     *       bsf 转换后的参数副本派生（annexb extradata）单独 open。
     */
    /* output_cuda: true = hybrid_gpu —— 输出帧驻留显存（GPU chunk
     * 零拷贝、CPU chunk H2D 上载）；false = 输出统一落 CPU。 */
    HybridThreadedDecoder(int device_id,
                          AVCodecParameters *codecpar,
                          const AVInputFormat *iformat,
                          bool output_cuda = false);
    ~HybridThreadedDecoder() override;

    void SetCodecContext(AVCodecContext *dec_ctx, int width = -1, int height = -1,
                         int rotation = 0, int output_format = 0) override;
    void SetRoi(int x1, int y1, int x2, int y2) override;
    void Start() override;
    void Stop() override;
    void Clear() override;
    void Push(ffmpeg::AVPacketPtr pkt, runtime::NDArray buf) override;
    bool Pop(runtime::NDArray *frame) override;
    bool Drained() const override;
    void SuggestDiscardPTS(std::vector<int64_t> dts) override;
    void ClearDiscardPTS() override;

    /*! \brief 供 VideoReader 在关键帧索引就绪后注入 (pts, rank) 表与
     *  总帧数 —— chunk 的 expected 帧数 = 相邻关键帧 rank 之差，
     *  是合并逻辑 "补满才关 chunk" 的基准。 */
    void SetKeyframeRanks(std::vector<int64_t> pts_list,
                          std::vector<int64_t> rank_list,
                          int64_t frame_count);

    /*! \brief 混合统计（诊断用）：两路各自输出的帧数 */
    int64_t FramesDecoded(bool gpu_side) const { return frames_out_[gpu_side ? 1 : 0].load(); }

  private:
    enum Side : int { SIDE_CPU = 0, SIDE_GPU = 1 };

    struct Chunk {
        Side side;
        int64_t start_pts;   ///< chunk 首帧（关键帧）pts
        int64_t end_pts;     ///< 下一关键帧 pts；最后一个 chunk 为 INT64_MAX
        int64_t expected;    ///< 该 chunk 应发射的帧数（kf rank 差；0=未知）
        int64_t emitted;     ///< 已发射帧数
        std::chrono::steady_clock::time_point first_pop;  ///< 首帧弹出时刻（测速）
        bool timed = false;
    };

    /*! \brief 重置路由/合并状态（Seek/Clear 时调用；保留速率 EWMA 与 kf 索引） */
    void ResetRouting();
    /*! \brief 为起始 pts == key_pts 的新 chunk 选择承接侧（速率感知贪心） */
    Side ChooseSide(int64_t key_pts);
    /*! rief pts 	o 呈现序帧号（kf 表近似，调度用） */
    int64_t RankOfPts(int64_t pts) const;
    /*! \brief chunk [start,end) 的期望帧数（查 kf 索引；0=未知） */
    int64_t ExpectedFrames(int64_t start_pts, int64_t end_pts) const;
    /*! \brief 该 codec 的关键帧是否 IDR 型（决定可否用 kick 冲刷/混合路由） */
    bool IsIdrLikeCodec() const;
    /*! \brief 取一帧（stash 优先 → CPU 子解码器 / GPU ready_ 队列） */
    bool PopSide(Side s, runtime::NDArray *f);
    /*! \brief GPU 帧搬到主机内存（布局两侧逐字节一致，整块 D2H） */
    static runtime::NDArray ToHost(const runtime::NDArray &gpu_frame);
#ifdef DECORD_USE_CUDA
    /*! \brief GPU 工作线程主循环：落地（LandStep）+ 喂包（FeedStep） */
    void GpuWorkerLoop();
    /*! \brief 落地一步：ready_ 有余量才 gpu_->Pop → D2H → ready_。
     *  持续排空 NVDEC 输出队列是显存有界的关键（reorder 无界堆积即
     *  7.7GB 峰值/OOM 的根因）。返回是否做了实际工作。 */
    bool LandStep();
    /*! \brief 喂包一步：ready_ 有余量才从 gpu_pkt_q_ 取包推给 GPU 侧。
     *  EOF 时在队列排空后补推 kMaxOutputSurfaces 个 flush 缓冲。
     *  返回是否做了实际工作。 */
    bool FeedStep();
    /*! \brief 上载一步（仅 GPU 驻留模式）：CPU 侧帧 H2D 入 cpu_ready_。
     *  先取池缓冲再取帧（帧不可回退入 CPU 解码器，必须不滞留）；
     *  marker 直通不拷贝。返回是否做了实际工作。 */
    bool UploadStep();
    /*! \brief 上载线程主循环（GPU 驻留模式） */
    void UploaderLoop();
    /*! \brief 上载专用非阻塞 CUDA 流。同步 H2D 若走 legacy 默认流，
     *  会与 CU 解码器 stream_（blocking 流）互斥 —— 每次上载都整体暂停
     *  NVDEC/转换管线（实测 hevc hybrid_gpu 被压到 0.75x）。 */
    void *up_stream_ = nullptr;
    /*! \brief pinned 暂存环（4 槽，仅上载线程访问）：pageable H2D
     *  走驱动内 staging（WDDM 下 ~1.2ms/帧），pinned H2D ~0.4ms —— 上载
     *  能力从 ~830fps 提到 ~1300fps+。 */
    void *up_staging_[4] = {nullptr, nullptr, nullptr, nullptr};
    std::size_t up_stage_bytes_ = 0;
    int up_stage_idx_ = 0;
    /*! \brief 停止并回收 GPU 工作线程（Stop/Clear 共用） */
    void StopGpuWorker();
#endif
    /*! \brief kInt64 drain marker 判定（与 NextFrameImpl 的判据一致） */
    static bool IsMarker(const runtime::NDArray &f);
    /*! \brief GPU 输出缓冲池形状（与 VideoReader::FrameShape 同语义） */
    std::vector<int64_t> GpuFrameShape() const;
    /* 输出形状（fmt: 0=RGB24, 1=GRAY8, 2=YUV420 packed 2D）：
     * yuv420 -> (h+ceil(h/2), w) 的 Y+交错UV；gray -> (h,w)；rgb -> (h,w,3)。
     * 与 VideoReader::FrameShape 同语义。 */
    static std::vector<int64_t> FrameShapeFor(int fmt, int h, int w);
    /*! \brief ready_ 队列的帧数上限（由字节预算换算，随分辨率自适应） */
    std::size_t ReadyCap() const;
    /*! \brief 记录 chunk 发射速率（EWMA，供 ChooseSide） */
    void RecordChunkRate(const Chunk &ch);

#ifdef DECORD_USE_CUDA
    /*! \brief GPU 输出缓冲池（有界，阻塞 Acquire）。声明在 gpu_ 之前：
     *  成员按声明逆序析构，保证 ~CUThreadedDecoder 销毁内部队列时
     *  在途缓冲经 deleter 回调的池仍然存活。 */
    HybridGpuBufferPool gpu_pool_;
    /* GPU 驻留模式专用：CPU chunk 上载帧的独立容器池。GPU 帧从解码到
     * 按序消费全程持有 gpu_pool_ 缓冲（周转=消费速率），若共用单池，
     * GPU 解码超前会占满全部缓冲、CPU 上载饿死（实测 19+9=28 卡死）。 */
    HybridGpuBufferPool up_pool_;
    std::unique_ptr<cuda::CUThreadedDecoder> gpu_;
    /*! \brief GPU 侧专属 codecpar 副本：CUThreadedDecoder 初始化 bsf
     * (mp4→annexb) 时会就地改写传入的 codecpar（extradata 变 annexb），
     * CPU 侧必须继续用原始 AVCC 参数 —— 两路各持一份，互不污染。 */
    ffmpeg::AVCodecParametersPtr gpu_codecpar_;
    /*! \brief 池大小：kMaxOutputSurfaces(20) + 解码/转换在途余量 */
    static constexpr std::size_t kGpuPoolBuffers = 28;
    /*! \brief GPU 驻留模式解码池：GPU 帧驻留显存、消费后才归还，池大小
     *  = NVDEC 超前上限 —— 28 帧会迫使 NVDEC 频繁停等消费（实测 hevc
     *  0.76x），扩到 128 帧让硬件解跑满。 */
    /*! GPU 驻留模式的解码池 = NVDEC 超前上限（帧驻留直到按序消费）。
     *  须覆盖一个 CPU chunk 的发射期 + 调度抖动（~286 帧 chunk → 384）。 */
    static constexpr std::size_t kGpuResidentPoolBuffers = 384;
    /*! \brief ready_ 落地队列字节预算（宿主 RAM 的硬边界） */
    /*! 宿主 RAM 预算（帧数 = 预算/frame_bytes）：GPU 解码超前的上限。
     *  chunk 交替时 NVDEC 需覆盖一个 CPU chunk 的发射期（~300 帧@1080p），
     *  1GiB 的 313 帧门控实测顶死（rdy 恒 320、包队列堆积 3000+）。
     *  3GiB ≈ 1000 帧@1080p。 */
    static constexpr std::size_t kReadyMaxBytes = 3ull << 30;
    /*! \brief GPU 工作线程（Start/Stop 管理）：喂包 + 落地 */
    std::thread lander_;
    /*! \brief CPU 帧上载线程（仅 GPU 驻留模式启动）。与落地分离：
     *  hevc NVDEC ~1800fps 时帧间 <0.6ms，若与 H2D 同步拷贝（~1.2ms/帧）
     *  同线程串行，GPU 落地节奏被上载拖住、NVDEC 被迫降速（实测 hevc
     *  hybrid_gpu 反而 0.71x 于纯 GPU）。 */
    std::thread uploader_;
    std::atomic<bool> lander_run_{false};
    /*! \brief 保护 gpu_pkt_q_ / gpu_flush_left_（demux 线程写，工作线程读） */
    mutable std::mutex lcv_mtx_;
    std::condition_variable lcv_;
    /*! \brief 待喂 GPU 包队列（宿主 RAM；demux Push 永不阻塞的关键） */
    std::deque<ffmpeg::AVPacketPtr> gpu_pkt_q_;
    /*! rief EOF flush 剩余数：>0 表示 EOF 已到、还有 flush 缓冲待喂
     *  （工作线程按池余量逐个推进，部分推进安全） */
    int gpu_flush_left_ = 0;
    /*! \brief 已落地待发射的宿主帧队列（含 GPU drain marker），字节预算有界 */
    mutable std::mutex rmtx_;
    std::deque<runtime::NDArray> ready_;
    /*! \brief GPU 驻留模式的落地预算（字节）：GPU 帧驻留显存直到按序
     *  消费，NVDEC 超前必须覆盖一个 CPU chunk 的发射期（~300 帧@1080p），
     *  1GiB 预算的 213 帧上限不够（实测 hevc 每 chunk 对损失 ~40ms）。 */
    static constexpr std::size_t kReadyMaxBytesGpu = 4ull << 30;
    /* GPU 驻留模式：CPU 侧已上载的显存帧队列（含 CPU drain marker），
     * 发射序保持。CPU 落地模式不用（CPU 帧直读子解码器）。 */
    std::deque<runtime::NDArray> cpu_ready_;
    /* 输出设备：false=落 CPU（hybrid），true=驻留显存（hybrid_gpu） */
    bool out_cuda_ = false;
    /*! rief GPU 侧实测产出速率（落地 EWMA，帧/秒）。调度用它而非
     *  chunk 发射速率：发射被"前序 chunk 消费"拖长，会把 NVDEC 能力
     *  低估数倍（实测 hevc GPU 被低估后 util 仅 17%）。 */
    std::atomic<double> gpu_rate_landed_{0.0};
    /*! rief 上一帧落地时刻（仅工作线程访问） */
    std::chrono::steady_clock::time_point last_land_tp_{};
    /*! rief 当前连续落地段的帧数/秒数（仅工作线程访问） */
    int64_t land_seg_frames_ = 0;
    double land_seg_secs_ = 0.0;
#endif
    /*! \brief GPU 待发射帧字节预算：超过即强制 CPU 承接新 chunk
     *  （调度失衡时 GPU 包队列与在途帧的总量上限；非 CUDA 构建不触发） */
    static constexpr int64_t kGpuAheadBytes = 16ll << 30;
    // 等效 16GiB ≈ 5160 帧@1080p。注意它限制的真实资源只是宿主
    // 压缩包队列（~17KB/帧 → 上限 ~90MB RAM）；显存（池 28 帧）与
    // ready_（1GiB）各有独立预算。预算过小会让冷启动的 GPU chunk
    // 立即顶格、此后全部强制 CPU（实测 13:5 分配锁死）。
    /*! \brief 已路由到 GPU、尚未发射/丢弃的帧数（包粒度精确计数；
     *  ChooseSide 的硬预算依据） */
    int64_t gpu_pending_ = 0;

    ffmpeg::FFMPEGThreadedDecoder cpu_;
    int device_id_;
    int width_ = -1, height_ = -1, rotation_ = 0, output_format_ = 0;
    AVCodecID codec_id_ = AV_CODEC_ID_NONE;
    /*! \brief 单帧宿主字节数（ready_ 预算换算用；0=未知） */
    int64_t frame_bytes_ = 0;

    // ── kf 索引（SetKeyframeRanks 注入；expected 帧数的基准）──
    std::vector<int64_t> kf_pts_;   ///< 关键帧 pts（升序）
    std::vector<int64_t> kf_rank_;  ///< 对应呈现序帧号
    int64_t frame_count_ = 0;
    double pts_per_frame_ = 0;  ///< pts/每帧（kf 表线性估计，速率统一帧/秒用）

    // ── 路由状态（仅 Push 调用线程访问：VideoReader 单线程 demux）──
    bool routing_active_ = false;   ///< 已见到首包
    Side cur_side_ = SIDE_CPU;      ///< 当前承接 chunk 的侧
    int64_t cur_start_pts_ = 0;
    /*! \brief 当前包按 pts 归属解析出的目标侧（仅 Push 线程访问） */
    Side cur_route_override_ = SIDE_CPU;

    // ── 合并状态（Push 与 Pop 并发访问，mutex 保护）──
    mutable std::mutex mtx_;
    std::deque<Chunk> emit_queue_;      ///< 按分配顺序待发射的 chunk
    runtime::NDArray stash_[2];         ///< 各侧越界帧暂存（kick 产物等）
    bool has_stash_[2] = {false, false};
    bool eof_pushed_ = false;

    // ── 速率感知（pts 单位/秒；跨 Clear 保留）──
    double rate_ewma_[2] = {0.0, 0.0};  ///< 各侧 chunk 发射速率 EWMA
    int64_t backlog_end_[2] = {0, 0};   ///< 各侧已分配到的 pts 上界
    int64_t emitted_upto_[2] = {0, 0};  ///< 各侧已发射到的 pts
    int chunks_assigned_[2] = {0, 0};
    int64_t emitted_total_ = 0;          ///< 全局已发射帧数（消费位置）
    int64_t side_pending_[2] = {0, 0};   ///< 各侧已路由未发射帧数（真积压，
                                         ///< 含在途解码与存货，包粒度精确）
    int64_t last_chunk_frames_ = 0;      ///< 上一 chunk 帧数（新 chunk 估计）

    std::vector<int64_t> gpu_frame_shape_;

    std::atomic<int64_t> frames_out_[2]{};
    std::atomic<bool> started_{false};
};

}  // namespace decord

#endif  // DECORD_VIDEO_HYBRID_THREADED_DECODER_H_
