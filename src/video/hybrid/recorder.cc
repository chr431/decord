// hybrid/recorder.cc —— 停滞飞行记录器（2026-09-28 R3-1 续：零侵扰取证）。
//
// 进程内取证的两难：打印会改时序（HYBRID_DEBUG 2 万行 stderr 让失败
// 率 60-75% 的竞态 4/4 不复现），不打印则失败现场零信息。本记录器用
// **专用诊断线程** 4Hz 快照关键状态写到 decord_hybrid_flight.log：
// - 消费者路径零接触（不打印、不加锁竞争热点、不碰队列操作）
// - 每次快照三把锁各持 ~µs 级、4Hz——对时序的扰动比一次 fprintf 低
//   数个量级
// - env 门控：DECORD_HYBRID_FLIGHT=1 才启动（默认零开销）
// - 锁耗时标记：任一锁获取 >0.2s / 快照总耗时 >0.5s 写 # LOCK-SLOW /
//   # SNAP-SLOW 行——「记录器活着却零采样」时指认持锁者（2026-09-28
//   av1 停滞轮：FATAL 自旋期间记录器停写，即此诊断的靶点）
#include "../hybrid_threaded_decoder.h"

#include "../ffmpeg/ffmpeg_common.h"
#include "../nvcodec/cuda_threaded_decoder.h"
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

namespace decord {

void HybridThreadedDecoder::FlightRecorderLoop() {
    FILE *f = fopen("decord_hybrid_flight.log", "a");
    if (!f) return;
    const auto t0 = std::chrono::steady_clock::now();
    fprintf(f, "# flight begin this=%p\n", static_cast<void *>(this));
    fflush(f);
    while (flight_run_.load(std::memory_order_relaxed)) {
        const auto tw0 = std::chrono::steady_clock::now();
        std::size_t eq = 0, gsz = 0;
        int hside = -2;
        int64_t hs = -1, hexp = -1, hem = -1, hpend = 0;
        bool eofp = false, armf = false, eofo = false;
        {
            const auto tl0 = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lk(mtx_);
            const double dl = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tl0).count();
            if (dl > 0.2) fprintf(f, "# LOCK-SLOW mtx %.3fs\n", dl);
            eq = emit_queue_.size();
            gsz = gops_.size();
            if (!emit_queue_.empty()) {
                const Chunk &c = emit_queue_.front();
                hside = static_cast<int>(c.side);
                hs = c.start_pts;
                hexp = c.expected;
                hem = c.emitted;
            }
            eofp = eof_pushed_;
            armf = arm_flush_sent_;
            eofo = eof_flush_out_;
            hpend = side_pending_[0] + side_pending_[1];
        }
        std::size_t crdy = 0, rdy = 0;
        {
            const auto tl0 = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lk(rmtx_);
            const double dl = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tl0).count();
            if (dl > 0.2) fprintf(f, "# LOCK-SLOW rmtx %.3fs\n", dl);
            crdy = cpu_ready_.size();
            rdy = ready_.size();
        }
        std::size_t qpk = 0;
        int64_t gfl = 0;
        {
            const auto tl0 = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lk(lcv_mtx_);
            const double dl = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tl0).count();
            if (dl > 0.2) fprintf(f, "# LOCK-SLOW lcv_mtx %.3fs\n", dl);
            qpk = gpu_pkt_q_.size();
            gfl = gpu_flush_left_;
        }
        int64_t cud_p = -1, cud_b = -1, cud_o = -1;
        if (gpu_) gpu_->DiagDepths(&cud_p, &cud_b, &cud_o);
        const double t = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double tw = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tw0).count();
        if (tw > 0.5) fprintf(f, "# SNAP-SLOW %.3fs (t=%.3f)\n", tw, t);
        fprintf(f,
                "%7.3f eq=%zu gops=%zu head=%d,%lld,%lld,%lld pend=%lld "
                "crdy=%zu rdy=%zu qpk=%zu gfl=%lld cq=%lld cp=%lld "
                "cud=%lld/%lld/%lld fc=%lld fg=%lld em=%lld fe=%lld "
                "eof=%d%d%d\n",
                t, eq, gsz, hside, (long long)hs, (long long)hexp,
                (long long)hem, (long long)hpend, crdy, rdy, qpk,
                (long long)gfl, (long long)cpu_.QueueDepth(),
                (long long)cpu_.PendingDepth(), (long long)cud_p,
                (long long)cud_b, (long long)cud_o,
                (long long)frames_out_[0].load(std::memory_order_relaxed),
                (long long)frames_out_[1].load(std::memory_order_relaxed),
                (long long)emitted_total_,
                (long long)force_eof_close_.load(std::memory_order_relaxed),
                (int)eofp, (int)armf, (int)eofo);
        fflush(f);
        for (int i = 0; i < 25 && flight_run_.load(std::memory_order_relaxed);
             ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    fprintf(f, "# flight end\n");
    fclose(f);
}


void HybridThreadedDecoder::DumpState(const char *tag) const {
    std::size_t eq = 0;
    int hside = -2;
    int64_t hs = -1, hexp = -1, hem = -1;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        eq = emit_queue_.size();
        if (!emit_queue_.empty()) {
            const Chunk &c = emit_queue_.front();
            hside = static_cast<int>(c.side);
            hs = c.start_pts; hexp = c.expected; hem = c.emitted;
        }
    }
    std::size_t crdy = 0, rdy = 0;
    { std::lock_guard<std::mutex> lk(rmtx_); crdy = cpu_ready_.size(); rdy = ready_.size(); }
    std::size_t qpk = 0; int64_t gfl = 0;
    { std::lock_guard<std::mutex> lk(lcv_mtx_); qpk = gpu_pkt_q_.size(); gfl = gpu_flush_left_; }
    int64_t cud_p = -1, cud_b = -1, cud_o = -1;
    if (gpu_) gpu_->DiagDepths(&cud_p, &cud_b, &cud_o);
    fprintf(stderr,
            "[dump@%s] eq=%zu head=%d,%lld,%lld,%lld pend=%d/%d "
            "crdy=%zu rdy=%zu qpk=%zu gfl=%lld cq=%lld cp=%lld "
            "cud=%lld/%lld/%lld "
            "fc=%lld fg=%lld em=%lld fe=%lld eof=%d%d%d drained=%d\n",
            tag, eq, hside, (long long)hs, (long long)hexp, (long long)hem,
            (int)side_pending_[0], (int)side_pending_[1], crdy, rdy, qpk,
            (long long)gfl, (long long)cpu_.QueueDepth(),
            (long long)cpu_.PendingDepth(),
            (long long)cud_p, (long long)cud_b, (long long)cud_o,
            (long long)frames_out_[0].load(std::memory_order_relaxed),
            (long long)frames_out_[1].load(std::memory_order_relaxed),
            (long long)emitted_total_,
            (long long)force_eof_close_.load(std::memory_order_relaxed),
            (int)eof_pushed_, (int)arm_flush_sent_, (int)eof_flush_out_,
            (int)Drained());
    fflush(stderr);
}

bool HybridThreadedDecoder::ArmStalled() const {
    return arm_stalled_.load(std::memory_order_acquire);
}

void HybridThreadedDecoder::ClearArmStall() {
    arm_stalled_.store(false, std::memory_order_release);
}

}  // namespace decord
