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
 * 总在途由 demux prefetch 窗口（消费驱动）+ ready_ 预算 + 池深三重
 * 硬上限兜底，调度失衡时压缩包队列不会无界增长。
 */

#ifndef DECORD_VIDEO_HYBRID_THREADED_DECODER_H_
#define DECORD_VIDEO_HYBRID_THREADED_DECODER_H_

#include "threaded_decoder_interface.h"
#include "ffmpeg/threaded_decoder.h"
#include "storage_pool.h"

#include <atomic>
#include <functional>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace decord {

#ifdef DECORD_USE_CUDA
/*!
 * \brief pinned 主机帧池（CPU-out 落地专用）：cudaHostAlloc 块经
 *  NDArray::FromRecycled 包装，D2H 直达最终帧 —— 替代"pinned 暂存 +
 *  每帧 Empty 分配 + memcpy"的旧三步落地（每帧 ~0.3ms memcpy +
 *  3.1MB 分配器churn）。池尽 = LandStep 背压（回退旧路径无需）。
 *  块由 deleter 归还复用；尺寸不匹配（ROI 重建）的旧块直接释放。
 */
class PinnedHostFramePool
        : public std::enable_shared_from_this<PinnedHostFramePool> {
  public:
    void Reset(std::size_t max_cap, std::size_t frame_bytes,
               std::vector<int64_t> shape);
    bool Acquire(runtime::NDArray *out);   ///< 非阻塞；false = 池尽/禁用
    bool Enabled() const { return max_cap_ > 0 && !disabled_; }
    std::size_t frame_bytes() const { return bytes_; }
    void MarkDisabled() { disabled_ = true; }

  private:
    struct Holder {
        std::shared_ptr<PinnedHostFramePool> pool;
        void *data;
        std::size_t bytes;
    };
    static void ReturnDeleter(runtime::NDArray::Container *self);
    void Return(void *data, std::size_t bytes);
    std::mutex mtx_;
    std::deque<void *> free_;
    std::size_t created_ = 0;
    std::size_t cap_ = 0;       ///< 当前允许的已建块数（耗尽翻倍）
    std::size_t max_cap_ = 0;
    std::size_t bytes_ = 0;
    bool disabled_ = false;
    std::vector<int64_t> shape_;
};
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
    /*! \brief 诊断快照（stall 归因）：池的空闲/已建块数（非 const：仅
     *  工作/消费线程在锁内读，避免把整池 mutex 改 mutable） */
    std::size_t DiagFree() {
        std::lock_guard<std::mutex> lk(mtx_);
        return free_.size();
    }
    std::size_t DiagCreated() {
        std::lock_guard<std::mutex> lk(mtx_);
        return created_;
    }
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
    std::size_t cap_ = 0;      ///< 当前允许的已建块数（耗尽翻倍）
    std::size_t max_cap_ = 0;  ///< 自适应预算上限
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
    };

    /*! \brief 重置路由/合并状态（Seek/Clear 时调用；保留速率 EWMA 与 kf 索引） */
    void ResetRouting();
    /*! \brief 为起始 pts == key_pts 的新 chunk 选择承接侧（速率感知贪心） */
    Side ChooseSide(int64_t key_pts);
    /*! \brief 诊断/实验开关 DECORD_HYBRID_FORCE_SIDE 的生效侧；未设时返回
     *  Side(-1)（哨兵，与本文件既有写法一致）。
     *
     *  **ChooseSide 与 BuildPlan 必须共用本函数**：预路由计划早先自己按
     *  贪心份额算侧、不查这个覆盖，于是 force=cpu 下仍会规划出 GPU chunk，
     *  而这类 chunk 永远不会被喂包 —— Pop 卡死在队头（实测
     *  `[pop-stall] side=1 crdy=637 rdy=0 head=(side1,…,exp=299,em=293)`：
     *  637 帧已上载的 CPU 存货被一个不会来的 GPU chunk 堵在后面），
     *  且 force-close 安全网因 side_pending_[GPU]≠0 而正确地不敢关 chunk。 */
    Side ForcedSide() const;
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
    /*! \brief 落地线程主循环：只做 LandStep（NVDEC 收帧 → D2H → ready_） */
    void GpuWorkerLoop();
    /*! \brief 喂包线程主循环：只做 FeedStep。与落地分离 —— FeedStep 极
     *  廉价但旧实现与 LandStep（含 3.1MB/帧 D2H 收割 memcpy）串行，落地
     *  1500fps 时单线程循环率 ~2200/s 已在 NVDEC 供包临界（1868/s），
     *  落地越忙喂包越饿，rg 被压到 ~1400（独跑 1868）。 */
    void FeederLoop();
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
     *  marker 直通前先冲刷在途环（保序）。返回是否做了实际工作。 */
    bool UploadStep();
    /*! \brief 上载线程主循环（GPU 驻留模式） */
    void UploaderLoop();
    /*! \brief 上载专用非阻塞 CUDA 流。同步 H2D 若走 legacy 默认流，
     *  会与 CU 解码器 stream_（blocking 流）互斥 —— 每次上载都整体暂停
     *  NVDEC/转换管线（实测 hevc hybrid_gpu 被压到 0.75x）。 */
    void *up_stream_ = nullptr;
    /*! \brief 上载批大小：≤up_batch_ 帧逐帧 pinned+async 提交，批末一次
     *  cudaStreamSynchronize 统一收割（摊销同步开销；宿主 memcpy 与
     *  H2D 在批内重叠）。运行期默认 8，`DECORD_HYBRID_UPLOAD_BATCH`
     *  可消融覆盖（1 = 逐帧同步旧行为）。
     *
     *  **批大小不是杠杆（2026-09-12 消融定论）**：有效批均 1.1-3 的约束
     *  是 cpu-empty（上传者消费快于 CPU 解码者补充，队列近恒空），不是
     *  上限——上限提到 32/64 批均纹丝不动；反向压到 1（逐帧同步旧行为）
     *  引擎全片 e2e 也无差（hevc 配对 3 遍中位 9.54 vs 9.66s，符号混乱
     *  = 漂移内；av1/h264 同）。上传同步成本不在关键路径，cpu-empty 提前
     *  冲刷（低发射延迟）是正确默认，勿再做上传链重构。 */
    static constexpr int kUploadBatch = 64;  ///< 数组容量上限（非行为默认）
    int up_batch_ = 8;                       ///< 运行期批大小（仅上载线程读写）
    /*! \brief pinned 暂存槽（每批帧各一槽，仅上载线程访问）：pageable
     *  H2D 走驱动内 staging（WDDM 下 ~1.2ms/帧），pinned H2D ~0.4ms。
     *  批内帧的槽到批末 sync 前都保持占用（H2D 源数据保活）。 */
    void *up_staging_[kUploadBatch] = {};
    std::size_t up_stage_bytes_ = 0;
    /*! \brief D2H 异步中转环（仅 CPU-out 模式，工作线程访问）：GPU 帧
     *  cudaMemcpyAsync 到 pinned 槽，滞后 kD2HRingSlots 帧收割（事件
     *  已远，零等待）+ memcpy 到宿主 NDArray 入 ready_。此前同步 D2H
     *  串行在 LandStep（每帧 ~0.5-1ms），GPU 侧落地被压到 ~1200fps。 */
    static constexpr int kD2HRingSlots = 8;
    void *d2h_staging_[kD2HRingSlots] = {};
    void *d2h_ev_[kD2HRingSlots] = {};          ///< cudaEvent_t
    runtime::NDArray d2h_host_[kD2HRingSlots];  ///< 目标宿主帧（保活）
    runtime::NDArray d2h_gpu_[kD2HRingSlots];   ///< 源 GPU 帧（事件前不回池）
    bool d2h_valid_[kD2HRingSlots] = {};
    bool d2h_pooled_[kD2HRingSlots] = {};  ///< host 帧来自 pinned 池（收割零拷贝）
    int64_t d2h_seq_ = 0;
    std::size_t d2h_bytes_ = 0;
    void *d2h_stream_ = nullptr;   ///< D2H 专用非阻塞流（与 NVDEC 流并行）
    /*! \brief pinned 主机帧池（CPU-out）：D2H 直达最终帧，消除暂存
     *  memcpy + 每帧 Empty 分配。池深随 ready 预算（自适应上限），
     *  池尽 = 背压；cudaHostAlloc 失败自动降级回暂存路径。 */
    std::shared_ptr<PinnedHostFramePool> pinned_pool_;
    /*! \brief 冲刷 D2H 环（marker 前保序 / Stop 前）：收割全部在途帧
     *  入 ready_。ready_ 满时容忍越界（软限，仅 EOF 尾部发生）。 */
    void FlushD2H();
    /*! \brief 收割一个 D2H 槽（ready 有余量为前提），失败返回 false */
    bool HarvestD2H(int k);
    /*! \brief 同步并丢弃在途 D2H/H2D（Clear/ROI 重建用） */
    void AbortInflight();
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
    /*! \brief 硬件自适应预算计算（空闲显存/内存 → 各池深/队列/prefetch） */
    void ComputeBudgets();
    double vram_budget_ = 768.0 * 1024 * 1024;  ///< 显存预算（ROI 重建池深复用）
    /*! \brief demux 领先深度建议（自适应预算计算结果）。
     *  盲阶段（双侧速率未就绪，sched_initialized_ 未置位）收缩到
     *  ~2 chunks：demux 全速领先（千余包瞬时入队）会让十几个 chunk
     *  的路由决策跑在速率学习之前 —— hevc 实测 13 chunks 全部
     *  rc-unknown 盲采 CPU（90% 帧量），water-filling 上线前流已
     *  耗尽（混跑 1042 vs 纯 GPU 1836）。收缩后决策随发射实时
     *  推进，首个 CPU-sample chunk 的段速率（16 帧首折）就位后
     *  即转入稳态全深。 */
    int SuggestPrefetchDepth() const override {
        if (!sched_initialized_) {
            const int est = static_cast<int>(est_chunk_frames_);
            const int blind = est > 0 ? 2 * est + 64 : 640;
            return std::min(prefetch_frames_, blind);
        }
        return prefetch_frames_;
    }
    /*! 重试推包门控：side_pending_（两侧在途帧，kick 计入、发射/陈旧
     *  丢弃逐帧核销）是精确在途 —— VideoReader 的包账目因 hybrid 侧
     *  丢弃永久虚高不可用（D3 教训），而重试无条件推包会让 demux 以
     *  消费轮询速度跑到解码前面（hevc 实测 10 决策/50ms 全部盲分
     *  CPU）。窗口必须 ≥ 两侧满库存之和（CPU queue + GPU ready +
     *  上载容器 + 池在途）：块交替下两侧库存都会顶到各自上限，窗口
     *  偏小会把 front chunk 后续的包拦死（发射顺序串行 → 死锁，
     *  hybrid_gpu hevc 实测 1500 帧处挂死）。盲阶段收缩到 ~2 chunks
     *  让路由决策随发射推进、速率学习先于决策就位。 */
    bool NeedsPackets() const override {
        if (eof_pushed_) return false;
        // 未清偿的跨侧债务必须**无条件放行**：慢消费者把在途窗口顶满时
        // demux 暂停，若离场侧仍有滞留帧（见 kick_side_ 注释），唯一解法
        // 是继续供包驱动克隆清偿；窗口闸在此拦包 = §16.1 第一层死锁
        //（实测：kick 只到 1/5、np=0、队头 em=293/300 永冻）。清偿后
        // 恢复窗口闸；护栏击穿（损坏码流）时保持放行直到 EOF 冲刷兜底。
        // Push 与 NeedsPackets 同在消费线程（VideoReader 单线程 demux），
        // 无竞态。
        if (kick_side_ != Side(-1)
                && side_pending_[kick_side_] - kick_cloned_ > 0) {
            return true;
        }
        const int64_t inflight = side_pending_[0] + side_pending_[1];
        if (!sched_initialized_) {
            const int est = static_cast<int>(est_chunk_frames_);
            // 盲窗曾试过 +up_pool_frames_（为 "sustained 倾斜计划下
            // cpu_ready_ 存货吃满盲窗" 的 gpu-out 死锁为解）——该病灶已由
            // kick 突发治本（本文件 KICK_BURST），而加宽本身让 av1 gpu-out
            // 盲阶段 CPU 采样存货更深、保序等待拉长，真交付 70%→10%
            //（3x 崩盘，消融矩阵）。回退。
            const int64_t blind = est > 0 ? 2 * est + 64 : 640;
            return inflight < std::min<int64_t>(
                       static_cast<int64_t>(prefetch_frames_)
                           + queue_frames_ + ready_cap_frames_ + 512,
                       blind);
        }
        return inflight < static_cast<int64_t>(prefetch_frames_)
                          + queue_frames_ + ready_cap_frames_ + 512;
    }

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
    /*! GPU 驻留模式解码池的**下限**：实际池深由空闲显存自适应
     *  （SetCodecContext 计算，有界、分配失败自动收缩转背压不 OOM）。 */
    static constexpr std::size_t kGpuResidentPoolBuffers = 128;
    /*! \brief ready_ 落地队列字节预算（宿主 RAM 的硬边界） */
    /*! 宿主 RAM 预算（帧数 = 预算/frame_bytes）：GPU 解码超前的上限。
     *  chunk 交替时 NVDEC 需覆盖一个 CPU chunk 的发射期（~300 帧@1080p），
     *  1GiB 的 313 帧门控实测顶死（rdy 恒 320、包队列堆积 3000+）。
     *  3GiB ≈ 1000 帧@1080p。 */
    static constexpr std::size_t kReadyMaxBytes = 1ull << 30;
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
    static constexpr std::size_t kReadyMaxBytesGpu = 1ull << 30;
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
    /*! \brief 已路由到 GPU、尚未发射/丢弃的帧数（包粒度精确计数；
     *  诊断用途） */
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
    /*! \brief 跨侧切换的**反馈式清偿**（2026-09-12 定稿；取代两代定长
     *  KICK_BURST=5/16 的"猜重排深度"设计——16 对 bf16 金字塔（实测
     *  重排深度 17）仍不够，调参只是压概率不是消除）：
     *
     *  切换时刻快照离场侧债务 `kick_owed_ = side_pending_[side]`（含
     *  kick IDR 克隆包）；此后每个非 key 包，只要债务未清偿
     *  （`side_pending_[side] − kick_cloned_ > 0`：切换前路由的帧仍有
     *  滞留——发射与陈旧丢弃都减 pending，克隆自身加 pending 故扣除）
     *  就克隆一份喂给离场侧，驱动其 DPB 交出滞留帧；**清偿由解码器
     *  实际产出闭环**，不依赖任何重排深度假设。解除/作废：
     *  · 清偿（pending−cloned ≤ 0）→ 解除武装，NeedsPackets 恢复窗口闸；
     *  · 下个 key 包 → 作废（正常流重排深度 ≪ chunk 长度，一个整 chunk
     *    的克隆仍不清偿 = 码流损坏；武装保持放行 demux 至 EOF，EOF 的
     *    null 冲刷是确定性全量排空，队头必然解冻）；
     *  · 护栏 KICK_CLONE_GUARD（64）→ 停止克隆防无限克隆（武装保持，
     *    走 EOF 兜底同上）。`DECORD_HYBRID_KICK_BURST` 语义随迁：
     *    0 = 关闭反馈克隆（消融对照 = 单包 kick 旧行为），>0 = 护栏值。
     *  历史：KICK_BURST=5（4ccf889）治 hevc gpu-out 倾斜计划死锁
     *  （kick=0 消融 4/4 挂）；=16（02c91e6）治 h264lg 宿主路径 ONNX
     *  慢消费环死锁（两层：突发被在途窗口截断 + 5 包冲不开 B 金字塔）。
     *  仅 Push/NeedsPackets（同一消费线程）读写，无需加锁。 */
    static constexpr int KICK_CLONE_GUARD = 64;
    int kick_guard_ = KICK_CLONE_GUARD;  ///< 构造期读 env（0=消融关闭，>0=护栏值）
    int kick_owed_ = 0;          ///< 切换时刻债务快照（诊断用；机制只看 pending−cloned）
    int kick_cloned_ = 0;        ///< 本次切换已克隆包数（同步加 pending）
    Side kick_side_ = Side(-1);  ///< 债务侧；Side(-1) = 未武装
    /*! \brief 当前包按 pts 归属解析出的目标侧（仅 Push 线程访问） */
    Side cur_route_override_ = SIDE_CPU;

    // ── 合并状态（Push 与 Pop 并发访问，mutex 保护）──
    mutable std::mutex mtx_;
    std::deque<Chunk> emit_queue_;      ///< 按分配顺序待发射的 chunk
    runtime::NDArray stash_[2];         ///< 各侧越界帧暂存（kick 产物等）
    bool has_stash_[2] = {false, false};
    bool eof_pushed_ = false;

    // ── 速率感知（跨 Clear 保留）──
    int chunks_assigned_[2] = {0, 0};
    int64_t assigned_frames_[2] = {0, 0};  ///< 各侧累计分账帧数（chunk 关闭
                                           ///< 时按 expected 累计；份额 governor 用）
    // ── chunk 预路由（2026-09-10）：双侧速率就绪即按能力比配额一次性
    // 规划全部未来 chunk 的侧；demux 一读到 keyframe 包即查表路由，
    // 两解码器持续有包（混跑从"交替"变"并发"）。plan_side_ 按 kf 索引
    // 对齐，-1 = 未规划（回退 pending 策略）。──
    std::vector<int> plan_side_;
    bool plan_ready_ = false;
    void BuildPlan(int64_t key_pts, double cpu_share);
    int64_t emitted_total_ = 0;          ///< 全局已发射帧数（消费位置）
    int64_t side_pending_[2] = {0, 0};   ///< 各侧已路由未发射帧数（真积压，
                                         ///< 含在途解码与存货，包粒度精确）
    int64_t est_chunk_frames_ = 0;       ///< chunk 帧数估计（份额累计用）
    bool sched_initialized_ = false;     ///< 双侧速率首次就绪后已重置 alloc

    std::vector<int64_t> gpu_frame_shape_;
    // ── 硬件自适应预算（SetCodecContext 计算；全部有界）──
    int gpu_pool_frames_ = 128;    ///< GPU 解码池深（显存，按空闲量自适应）
    int up_pool_frames_ = 96;      ///< 上载池深（显存，GPU 驻留模式）
    int queue_frames_ = 384;       ///< CPU 存货队列深（RAM）
    int ready_cap_frames_ = 341;   ///< ready_ 帧数上限（RAM/显存口径合一）
    int prefetch_frames_ = 384;    ///< demux 领先深度建议（包）

    std::atomic<int64_t> frames_out_[2]{};
    std::atomic<bool> started_{false};
    // ── 非打印测量（2026-09-12，§7.2 先量后做）──────────────────────
    // 热路径只做 relaxed 整数累加，**不打 stderr**；唯一输出点是 Stop()
    // 里 `DECORD_HYBRID_STATS` 门控的一次性汇总块（析构时触发，不在被测
    // 墙钟窗口内）。对照物：DECORD_HYBRID_DEBUG 一次运行 13 万行打印，
    // 墙钟 2940 vs 3100fps —— 打印本身扰动测量，本轮起归因只用这里。
    //
    // HOL 记账口径：dt 累加发生在消费者线程的相邻 Pop 调用之间，因此
    // "head 空等"只在 decode-only 消费（Pop 背靠背）时等于真实墙钟损失；
    // 消费者中间干别的活（OCR）时该 dt 含消费工作时间，只能作上限读。
    std::atomic<int64_t> hol_us_[2]{};          ///< 队头阻塞时长 µs（按 head 侧分桶）
    std::atomic<int64_t> hol_ev_[2]{};          ///< 阻塞 episode 进入次数（非阻塞→阻塞）
    std::atomic<int64_t> hol_strand_max_[2]{};  ///< episode 期间观测到的对侧存货峰值
    int hol_prev_side_ = -1;  ///< 上次 Pop 结束时是否停在 HOL 阻塞（仅消费者线程读写）
    std::chrono::steady_clock::time_point hol_tp_{};
    std::atomic<int64_t> up_flush_n_{0}, up_flush_f_{0};  ///< UploadStep 批次数 / 批内帧数
    std::atomic<int64_t> up_nobuf_{0}, up_cempty_{0};     ///< 池尽 / CPU 断流提前冲刷
    std::atomic<int64_t> plan_rc_{0}, plan_rg_{0};        ///< BuildPlan 冻结时的两侧速率
    std::atomic<int64_t> plan_frames_[2]{};               ///< BuildPlan 各侧规划帧量
    std::atomic<int64_t> plan_folds_{0};                  ///< BuildPlan 时 CPU 已完成折数
    std::atomic<int64_t> kicks_[2]{};                     ///< 各侧收到的 kick 冲刷包数
    std::atomic<int64_t> fb_clones_{};                    ///< 反馈式清偿累计克隆包数
    std::atomic<int64_t> stats_t0_us_{0};                 ///< Push 首包时刻（steady epoch µs）
    std::atomic<int64_t> plan_age_ms_{0};                 ///< BuildPlan 距首包毫秒数
};

}  // namespace decord

#endif  // DECORD_VIDEO_HYBRID_THREADED_DECODER_H_
