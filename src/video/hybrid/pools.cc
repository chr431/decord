// hybrid/pools.cc —— HybridThreadedDecoder 实现 TU（R3-3 拆分，
// 2026-09-28 自 hybrid_threaded_decoder.cc 机械搬运；公共语义见
// ../hybrid_threaded_decoder.h，职责：设备侧：落地/D2H 环/喂包/上载/工作线程（pools））
#include "../hybrid_threaded_decoder.h"

#include "../ffmpeg/ffmpeg_common.h"
#include <decord/runtime/ndarray.h>

#include "../nvcodec/cuda_threaded_decoder.h"

#include <dmlc/logging.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "../../runtime/cuda/cudart_shim.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstring>
#include <vector>

namespace decord {

runtime::NDArray HybridThreadedDecoder::ToHost(const runtime::NDArray &g) {
    CHECK(g.defined()) << "ToHost on undefined frame";
    const DLTensor *gtp = g.operator->();
    const DLTensor &gt = *gtp;
    runtime::NDArray h = runtime::NDArray::Empty(
        std::vector<int64_t>(gt.shape, gt.shape + gt.ndim), kUInt8, kCPU);
    h.pts = g.pts;
    // 三种输出格式 (rgb/gray/yuv420) 两侧布局逐字节一致：本 fork 的
    // yuv420 输出统一为 (h+ceil(h/2), w) 的 Y + 交错 UV 行（CPU filter 搬运
    // 与 GPU improc 直出的 packed 布局相同，实测逐字节相等）—— 无需重排，
    // 整块 D2H 即可。
    runtime::NDArray::CopyFromTo(const_cast<DLTensor *>(gtp),
                                 const_cast<DLTensor *>(h.operator->()), nullptr);
    return h;
}


void HybridThreadedDecoder::GpuFrameProduced() {
    // 生产侧 sustained 折（2026-09-18）：连续产出段（间隔 <50ms，
    // 断流重置——CPU 值日期不计入，"没活干"≠"能力低"）满折（16 帧）
    // 入 8 折环，出版 max(环内"连续4折最小值") + 0.997/折慢降锁
    // （对齐 cpu_.prod_rate_ 唯一口径）。旧落地侧 EWMA 两病（排空
    // 积压假折 689655fps / 冷启排空孤峰直接出版 13402fps → CPU 臂
    // 超分）同源于"量到消费节奏"——生产侧不受积压/消费影响。
    const auto now = std::chrono::steady_clock::now();
    if (last_land_tp_.time_since_epoch().count() != 0) {
        double dt = std::chrono::duration<double>(now - last_land_tp_).count();
        if (dt > 1e-6 && dt < 0.05) {
            land_seg_frames_++;
            land_seg_secs_ += dt;
            if (land_seg_frames_ >= 16) {
                const double r = land_seg_frames_ / land_seg_secs_;
                land_fold_ring_[land_fold_i_] = r;
                land_fold_i_ = (land_fold_i_ + 1) % kLandFoldRing;
                if (land_fold_n_ < kLandFoldRing) ++land_fold_n_;
                if (land_fold_n_ >= 4) {
                    double best = 0.0;
                    const int oldest =
                        (land_fold_i_ - land_fold_n_ + 2 * kLandFoldRing)
                        % kLandFoldRing;
                    for (int i = 0; i + 4 <= land_fold_n_; ++i) {
                        double w = land_fold_ring_[
                            (oldest + i) % kLandFoldRing];
                        for (int j = 1; j < 4; ++j) {
                            const double v = land_fold_ring_[
                                (oldest + i + j) % kLandFoldRing];
                            if (v < w) w = v;
                        }
                        if (w > best) best = w;
                    }
                    const double prev = gpu_rate_landed_.load(
                        std::memory_order_relaxed);
                    gpu_rate_landed_.store(
                        prev > 0 ? std::max(best, prev * 0.997) : best,
                        std::memory_order_relaxed);
                }
                land_seg_frames_ = 0;
                land_seg_secs_ = 0.0;
            }
        } else {
            land_seg_frames_ = 0;
            land_seg_secs_ = 0.0;
        }
    }
    last_land_tp_ = now;
}

bool HybridThreadedDecoder::LandStep() {
    if (!gpu_) return false;
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        if (ready_.size() >= ReadyCap()) return false;  // 预算内背压
    }
    // 预取 pinned 池帧（CPU-out 直达 D2H 用）：必须在 gpu_->Pop 之前 ——
    // pop 之后无法把帧塞回解码器。池尽**不背压而是降级**：走下方暂存
    // 路径（慢但永不枯竭）—— 若池尽返回 false 会停止 pop，而 ready 未
    // 满时 FeedStep 仍在喂包，NVDEC 64 槽事件环随即溢出 FATAL（解码
    // 线程死亡 = 永久挂等；lockstep 慢消费实测）。ready 闸是唯一背压，
    // 与 FeedStep 的喂包闸同口径对齐，不会过喂。
    runtime::NDArray pooled;
    if (!out_cuda_ && pinned_pool_ && pinned_pool_->Enabled()) {
        pinned_pool_->Acquire(&pooled);  // 失败 = 本次走暂存路径
    }
    runtime::NDArray f;
    if (!gpu_->Pop(&f) || !f.defined()) return false;
    // 产出速率统计已外迁 GpuFrameProduced（CU display 线程逐帧触
    // 发，生产侧口径 2026-09-18）：落地线程排空 reorder 积压时会以
    // D2H 提交速度连 pop（假折 689655fps），生产侧不受积压影响；
    // GPU 驻留模式逐帧回调亦天然解除旧"统计挂 ToHost 后 rg 恒 0"坑。
    if (IsMarker(f)) {
        // drain marker（kCPU kInt64）：在途 D2H 帧必须先于 marker 入队
        // （发射顺序 = ready_ 顺序）。ready_ 满时容忍越界（软限，仅
        // EOF 尾部发生一次）。
        FlushD2H();
        std::lock_guard<std::mutex> lk(rmtx_);
        ready_.push_back(std::move(f));
        return true;
    }
    if (out_cuda_) {
        // GPU 驻留模式：帧留在显存直通入队（零拷贝），随消费归还池
        std::lock_guard<std::mutex> lk(rmtx_);
        ready_.push_back(std::move(f));
        return true;
    }
    // 异步 D2H：提交到 pinned 中转环，滞后 kD2HRingSlots 帧收割
    // （事件早已完成，零等待）。GPU 源帧保活在环内，收割后回池。
    {
        const int k = static_cast<int>(d2h_seq_ & (kD2HRingSlots - 1));
        if (d2h_valid_[k] && !HarvestD2H(k)) return false;  // ready 满，稍后重试
        if (d2h_staging_[k] == nullptr
                || d2h_bytes_ != static_cast<std::size_t>(frame_bytes_)) {
            if (d2h_staging_[k] != nullptr) cudaFreeHost(d2h_staging_[k]);
            d2h_staging_[k] = nullptr;
            if (cudaMallocHost(&d2h_staging_[k],
                               static_cast<size_t>(frame_bytes_)) == cudaSuccess) {
                d2h_bytes_ = static_cast<std::size_t>(frame_bytes_);
            }
            if (d2h_ev_[k] == nullptr) {
                cudaEvent_t ev = nullptr;
                if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)
                        == cudaSuccess) {
                    d2h_ev_[k] = ev;
                }
            }
            if (d2h_stream_ == nullptr) {
                cudaStreamCreateWithFlags(
                    reinterpret_cast<cudaStream_t *>(&d2h_stream_),
                    cudaStreamNonBlocking);
            }
        }
        const char *src = static_cast<const char *>
            (const_cast<DLTensor *>(f.operator->())->data);
        // pinned 池路径：D2H 直达最终帧（收割零拷贝入 ready_，消除
        // 暂存 memcpy + 每帧 Empty 分配）。pooled 帧在 LandStep 入口
        // （gpu_->Pop 之前）预取 —— pop 之后无法把帧塞回解码器，池尽
        // 若在此时才失败 = 已弹出的帧被丢弃 = 永久缺帧死等（lockstep
        // 慢消费实测挂死）。入口预取失败 = 干净背压，帧留在解码器内。
        if (pooled.defined()) {
            void *dst = const_cast<DLTensor *>(pooled.operator->())->data;
            if (d2h_stream_ != nullptr && d2h_ev_[k] != nullptr) {
                pooled.pts = f.pts;
                cudaMemcpyAsync(dst, src, static_cast<size_t>(frame_bytes_),
                                cudaMemcpyDeviceToHost,
                                reinterpret_cast<cudaStream_t>(d2h_stream_));
                cudaEventRecord(reinterpret_cast<cudaEvent_t>(d2h_ev_[k]),
                                reinterpret_cast<cudaStream_t>(d2h_stream_));
                d2h_host_[k] = std::move(pooled);
                d2h_gpu_[k] = f;    // 源保活（事件前不回池）
                d2h_pooled_[k] = true;
                d2h_valid_[k] = true;
                ++d2h_seq_;
                return true;
            }
            // 无流/事件：池块上同步 D2H（罕见兜底），直接交付
            pooled.pts = f.pts;
            cudaMemcpy(dst, src, static_cast<size_t>(frame_bytes_),
                       cudaMemcpyDeviceToHost);
            std::lock_guard<std::mutex> lk(rmtx_);
            ready_.push_back(std::move(pooled));
            return true;
        }
        if (d2h_staging_[k] != nullptr && d2h_ev_[k] != nullptr
                && d2h_stream_ != nullptr) {
            cudaMemcpyAsync(d2h_staging_[k], src,
                            static_cast<size_t>(frame_bytes_),
                            cudaMemcpyDeviceToHost,
                            reinterpret_cast<cudaStream_t>(d2h_stream_));
            cudaEventRecord(reinterpret_cast<cudaEvent_t>(d2h_ev_[k]),
                            reinterpret_cast<cudaStream_t>(d2h_stream_));
        } else {
            // pinned/event 分配失败：退回同步 D2H（必然完成）
            runtime::NDArray h = ToHost(f);
            std::lock_guard<std::mutex> lk(rmtx_);
            ready_.push_back(std::move(h));
            return true;
        }
        const DLTensor &ft = *f.operator->();
        d2h_host_[k] = runtime::NDArray::Empty(
            std::vector<int64_t>(ft.shape, ft.shape + ft.ndim), kUInt8, kCPU);
        d2h_host_[k].pts = f.pts;
        d2h_gpu_[k] = f;    // 源保活（D2H 完成前不回池）
        d2h_valid_[k] = true;
        ++d2h_seq_;
    }
    return true;
}


bool HybridThreadedDecoder::HarvestD2H(int k) {
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        if (ready_.size() >= ReadyCap()) return false;  // 预算内背压
    }
    cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(d2h_ev_[k]));
    if (d2h_pooled_[k]) {
        // pinned 池帧：D2H 已直达，零拷贝入 ready_
        std::lock_guard<std::mutex> lk(rmtx_);
        ready_.push_back(std::move(d2h_host_[k]));
    } else {
        std::memcpy(d2h_host_[k].operator->()->data, d2h_staging_[k],
                    static_cast<size_t>(frame_bytes_));
        {
            std::lock_guard<std::mutex> lk(rmtx_);
            ready_.push_back(std::move(d2h_host_[k]));
        }
    }
    d2h_pooled_[k] = false;
    d2h_host_[k] = runtime::NDArray();
    d2h_gpu_[k] = runtime::NDArray();  // 源回池
    d2h_valid_[k] = false;
    return true;
}


void HybridThreadedDecoder::FlushD2H() {
    if (d2h_seq_ == 0) return;
    const int64_t base = d2h_seq_ - kD2HRingSlots;
    for (int64_t i = base; i < d2h_seq_; ++i) {
        if (i < 0) continue;
        const int k = static_cast<int>(i & (kD2HRingSlots - 1));
        if (!d2h_valid_[k]) continue;
        // marker 前的保序冲刷：ready_ 越界容忍（软限）；Stop 时放弃等待
        while (lander_run_.load() && !HarvestD2H(k)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!d2h_valid_[k]) continue;
        // 已停止：同步事件后丢弃（Clear/Stop 路径状态全清）
        cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(d2h_ev_[k]));
        d2h_host_[k] = runtime::NDArray();
        d2h_gpu_[k] = runtime::NDArray();
        d2h_valid_[k] = false;
    }
    d2h_seq_ = 0;
}


void HybridThreadedDecoder::AbortInflight() {
    // Clear/ROI 重建：同步在途拷贝（缓冲安全释放）后丢弃。
    // H2D 批量上载无跨 UploadStep 的在途状态（所有路径批内 sync 完成后
    // 才返回），上载线程 join 后此流必空闲 —— 防御性 sync 一次。
    for (int k = 0; k < kD2HRingSlots; ++k) {
        if (d2h_valid_[k]) {
            if (d2h_ev_[k] != nullptr) {
                cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(d2h_ev_[k]));
            }
            d2h_host_[k] = runtime::NDArray();
            d2h_gpu_[k] = runtime::NDArray();
            d2h_valid_[k] = false;
        }
    }
    d2h_seq_ = 0;
    if (up_stream_ != nullptr) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(up_stream_));
    }
}


bool HybridThreadedDecoder::FeedStep() {
    if (!gpu_) return false;
    bool has_pkt = false;
    ffmpeg::AVPacketPtr pkt;
    int flush_left = 0;
    {
        std::lock_guard<std::mutex> lk(lcv_mtx_);
        if (!gpu_pkt_q_.empty()) {
            pkt = std::move(gpu_pkt_q_.front());
            gpu_pkt_q_.pop_front();
            has_pkt = true;
        } else if (gpu_flush_left_ > 0) {
            flush_left = gpu_flush_left_;
        } else {
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> lk(rmtx_);
        // ready_ 接近上限即暂停喂包，越界的包留在宿主队列（RAM）。
        // GPU 驻留模式：ready_ 帧本身持有池缓冲，物理上限 = 池深
        // （预留 NVDEC 在途 surface）；旧公式把整个池重复计入
        // （ready+pool ≥ ReadyCap → ready 到 ReadyCap-pool 即停喂），
        // CPU 块发射期 NVDEC 有效产出被压到 ~1200fps（hevc/h264 混跑
        // 的主要损耗源）。CPU-out：ready_ 是宿主帧，池缓冲仅在途
        // （D2H 环 + surface ≈ kGpuPoolBuffers），上限仍是 ReadyCap。
        {
            const std::size_t cap = out_cuda_
                ? static_cast<std::size_t>(std::max(gpu_pool_frames_, 1))
                : ReadyCap();
            const std::size_t reserve = out_cuda_
                ? static_cast<std::size_t>(
                      ThreadedDecoderInterface::kMaxOutputSurfaces + 8)
                : kGpuPoolBuffers;
            if (ready_.size() + reserve >= cap) {
                std::lock_guard<std::mutex> lk2(lcv_mtx_);
                if (has_pkt) gpu_pkt_q_.push_front(std::move(pkt));
                return false;
            }
        }
    }
    if (!has_pkt) {
        // EOF flush：每 surface 一个 flush 缓冲（与 VideoReader 的 GPU
        // EOF 语义一致，显示回调永远能取到缓冲而不阻塞）。部分推进安全：
        // 首个 null 即触发 ENDOFSTREAM，其余 buf 只是补足配对。
        int fed = 0;
        while (fed < flush_left) {
            runtime::NDArray buf;
            if (!gpu_pool_.Acquire(&buf)) break;  // 池耗尽，下轮再试
            gpu_->Push(nullptr, buf);
            ++fed;
        }
        std::lock_guard<std::mutex> lk(lcv_mtx_);
        gpu_flush_left_ = flush_left - fed;
        return fed > 0;
    }
    runtime::NDArray buf;
    if (!gpu_pool_.Acquire(&buf)) {
        // 池耗尽：包放回队首，等 LandStep 回收后重试（不阻塞工作线程）
        std::lock_guard<std::mutex> lk(lcv_mtx_);
        gpu_pkt_q_.push_front(std::move(pkt));
        return false;
    }
    gpu_->Push(std::move(pkt), buf);
    return true;
}


bool HybridThreadedDecoder::UploadStep() {
    if (!out_cuda_) return false;
    // ── H2D 批量上载（重设计，替代每帧 sync）──
    // 旧实现每帧 memcpy 到 pinned + cudaMemcpyAsync 提交后立即
    // cudaStreamSynchronize（~0.4-0.7ms 串行）—— 上载能力 ~1400fps，
    // CPU chunk 的显存存货（up_pool）永远填不满（no-buf 1727 次/3000
    // 帧），CPU 块发射退化为上载实时跟随。批次化：≤kUploadBatch 帧逐
    // 帧 memcpy 到各自 pinned 槽 + async 提交（驱动流水化：H2D(i) 与
    // 宿主 memcpy(i+1) 重叠），批末一次 sync 统一收割。
    // 语义与逐帧同步完全一致：帧只在 H2D 完成后进 cpu_ready_、顺序 =
    // 提交顺序；marker/池尽/CPU 断流等所有提前退出路径先冲刷批再处理
    //（此前事件环方案的 marker/EOF 语义腐坏在这里不存在 —— 冲刷在
    // UploadStep 栈内同步完成，无跨调用在途状态）。
    // ── 积攒窗（2026-09-20 H2D 聚合轮）──
    // 批次化上线后实测批均仍只有 ~2.4 帧：CPU 臂断流（cpu-empty）即
    // 冲刷，UPLOAD_BATCH 8→64 消融无差的根因就是实际批从未变大——
    // 7761 帧仍是 7761 次小 H2D 提交，与 TRT 提交同上下文错叠，infer
    // p50 8.6→14.1ms（回到 trt_call 串行价，引擎侧启动轮 §10）。积攒
    // 窗：断流时不立即冲刷，窗内等续流攒大批（批均≈rc×W），一次 sync
    // 收割。只改发射时机，不改任何路径语义：marker/EOF/池尽仍立即
    // 冲刷；等待期释放池槽不占显存存货；稳态吞吐不变（恒定 W 只是
    // 一次性延迟，合并以 chunk/GOP 粒度保序，8 帧级突发无感）。
    static const bool dbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
    // 批大小消融旋钮（一次性读 env；仅上载线程读写 up_batch_，无锁）。
    static const int batch_env = [] {
        const char *e = getenv("DECORD_HYBRID_UPLOAD_BATCH");
        return e ? atoi(e) : 0;
    }();
    if (batch_env > 0) {
        up_batch_ = batch_env < kUploadBatch ? batch_env : kUploadBatch;
    }
    // 积攒窗宽度 µs（一次性读 env；默认 0 = 关）。2026-09-20 A/B
    // （h264/hevc hybrid 全片，臂序轮转）：机制生效（q_put_block −23%、
    // consume_feed −6% 双显著）但 wall 无净收益——h264 侧 decode.batch
    // +3.7%（HOL 延迟对冲），且当日 infer 暴露未复现（Σinfer 1.31s 已
    // = cpu 路径水平），无收益可兑现。默认关；机制保留供「暴露复现的
    // 机器状态」下复评（引擎侧启动轮 §10 的归因未获干预实验支持）。
    static const int wait_us = [] {
        const char *e = getenv("DECORD_HYBRID_UPLOAD_WAIT_US");
        return e ? atoi(e) : 0;
    }();
    runtime::NDArray bufs[kUploadBatch];
    int nf = 0;
    bool did = false;
    const auto batch_t0 = std::chrono::steady_clock::now();
    auto flush = [&]() {
        if (nf == 0) return;
        up_flush_n_.fetch_add(1, std::memory_order_relaxed);
        up_flush_f_.fetch_add(nf, std::memory_order_relaxed);
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(up_stream_));
        {
            std::lock_guard<std::mutex> lk(rmtx_);
            for (int i = 0; i < nf; ++i) {
                cpu_ready_.push_back(std::move(bufs[i]));  // pts 提交时已设
            }
        }
        if (dbg) fprintf(stderr, "[hybrid-u] flush %d", nf);
        nf = 0;
        did = true;
    };
    while (nf < up_batch_) {
        runtime::NDArray buf;
        if (!up_pool_.Acquire(&buf)) {
            up_nobuf_.fetch_add(1, std::memory_order_relaxed);
            if (dbg) fprintf(stderr, "[hybrid-u] no-buf");
            flush();  // 上载池耗尽：CPU 帧显存容器已满（背压）
            return did;
        }
        runtime::NDArray f;
        if (!cpu_.Pop(&f) || !f.defined()) {
            // 积攒窗：批内已有帧且窗未到期 → 释放池槽、小睡重试（EOF
            // marker 在队列尾，最长 250µs 后即被此重试取到并立即冲刷，
            // 尾部无额外等待）。
            if (nf > 0 && wait_us > 0) {
                const auto waited =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - batch_t0)
                                .count();
                if (waited < wait_us) {
                    buf = runtime::NDArray();   // 等待期不占池槽
                    std::this_thread::sleep_for(
                            std::chrono::microseconds(250));
                    continue;
                }
            }
            up_cempty_.fetch_add(1, std::memory_order_relaxed);
            if (dbg) fprintf(stderr, "[hybrid-u] cpu-empty");
            flush();  // buf 随析构归还池
            return did;
        }
        if (IsMarker(f)) {
            // marker 保序：批内帧全部 H2D 完成后才入队 marker
            flush();
            std::lock_guard<std::mutex> lk(rmtx_);
            cpu_ready_.push_back(std::move(f));
            return true;
        }
        const int k = nf;
        if (up_staging_[k] == nullptr
                || up_stage_bytes_ != static_cast<std::size_t>(frame_bytes_)) {
            if (up_staging_[k] != nullptr) {
                cudaFreeHost(up_staging_[k]);
                up_staging_[k] = nullptr;
            }
            if (cudaMallocHost(&up_staging_[k],
                               static_cast<size_t>(frame_bytes_)) == cudaSuccess) {
                up_stage_bytes_ = static_cast<std::size_t>(frame_bytes_);
            }
        }
        const char *src = static_cast<const char *>
            (const_cast<DLTensor *>(f.operator->())->data);
        char *dst_dev = static_cast<char *>
            (const_cast<DLTensor *>(buf.operator->())->data);
        if (up_staging_[k] != nullptr) {
            // 槽 k 占用至批末 sync（H2D 源保活）；宿主帧 f 的 memcpy 在
            // 提交前同步完成，无需保活。
            std::memcpy(up_staging_[k], src, static_cast<size_t>(frame_bytes_));
            cudaMemcpyAsync(dst_dev, up_staging_[k],
                            static_cast<size_t>(frame_bytes_),
                            cudaMemcpyHostToDevice,
                            reinterpret_cast<cudaStream_t>(up_stream_));
        } else {
            // pinned 分配失败的兜底：pageable 直接异步（驱动 staging，
            // 慢但正确 —— 批末 sync 同样保证完成）
            cudaMemcpyAsync(dst_dev, src, static_cast<size_t>(frame_bytes_),
                            cudaMemcpyHostToDevice,
                            reinterpret_cast<cudaStream_t>(up_stream_));
        }
        buf.pts = f.pts;
        bufs[nf] = std::move(buf);
        ++nf;
    }
    flush();
    return true;
}




void HybridThreadedDecoder::GpuWorkerLoop() {
    static const bool dbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
    while (lander_run_.load()) {
        bool did = false;
        try {
            did = LandStep();
            did = FeedStep() || did;
        } catch (const std::exception &e) {
            TrapWorkerError("gpu-worker", e);
            lcv_.notify_all();   // 唤醒消费者：Pop 处转 DECORDError
            break;               // 退出线程（不再 rethrow=不再 terminate）
        }
        if (dbg && did) fprintf(stderr, "[hybrid-w] did=%d ready=%zu cpuready=%zu pktq=%zu\n", (int)did, ready_.size(), cpu_ready_.size(), gpu_pkt_q_.size());
        if (!did) {
            if (dbg) fprintf(stderr, "[hybrid-w] idle rdy=%zu crdy=%zu q=%zu",
                             ready_.size(), cpu_ready_.size(), gpu_pkt_q_.size());
            std::unique_lock<std::mutex> lk(lcv_mtx_);
            lcv_.wait_for(lk, std::chrono::milliseconds(1));
        }
    }
}


void HybridThreadedDecoder::UploaderLoop() {
    while (lander_run_.load()) {
        bool did = false;
        try {
            did = UploadStep();
        } catch (const std::exception &e) {
            TrapWorkerError("uploader", e);
            lcv_.notify_all();
            break;
        }
        if (!did) {
            std::unique_lock<std::mutex> lk(lcv_mtx_);
            lcv_.wait_for(lk, std::chrono::milliseconds(1));
        }
    }
}


}  // namespace decord
