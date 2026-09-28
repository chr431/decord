// hybrid/pump.cc —— HybridThreadedDecoder 实现 TU（R3-3 拆分，
// 2026-09-28 自 hybrid_threaded_decoder.cc 机械搬运；公共语义见
// ../hybrid_threaded_decoder.h，职责：包缓存/供料泵/GOP 记账/消费侧 Pop（pump））
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
#include <string>
#include <cstring>
#include <vector>

namespace decord {

void HybridThreadedDecoder::PumpFeed(bool force_head) {
    // 供料泵（层0/1 重设计核心）：Push 只入缓存，这里在**分配时点**把
    // 包喂给臂。仅 Push 调用线程（VideoReader 单线程 demux）执行，
    // 沿用"路由状态单线程"不变量；每包一次 mtx_。
    //
    // 严格游标序供料：只供 cursor 指向的 GOP，供完才前进。曾试两档
    // 乱序（全乱序/同侧保序+跨侧前瞻）：跨侧乱序会让换侧 kick 帧失去
    // 流序相邻性（离场侧缺参考 → ffmpeg RPS 丢帧 → pending 泄漏挂死，
    // 实测 hevc 15k 帧 "Could not find ref with POC"）；严格序下供料序
    // =流序=分配序，换侧恒相邻，kick 语义与旧设计逐字一致。代价：
    // cursor 侧银行满时整泵暂停（对侧靠深库存撑住，利用率损失有界）。
    //
    // EOF（eof_cache_）：无视饥饿门排空全部剩余包——否则 demux 停读后
    // 无人再驱动泵，尾死锁。
    // flush 已实际发出（eof_flush_out_）：两臂已进 drain——此后任何喂包
    // 都是 send-after-flush 错误（2026-09-19 test6_h264 w12000 实测
    // ffmpeg CHECK）。Pop 的 force 重驱也必须止步于此。
    if (eof_flush_out_) {
        return;
    }
    for (int guard = 0; guard < 1000000; ++guard) {
        Side s = Side(-1);
        ffmpeg::AVPacketPtr pkt;
        bool is_key = false;
        ffmpeg::AVPacketPtr kick_pkt;   // 即发 kick 的克隆包（扫描时备好）
        bool clone_debt = false;
        Side debt_dst = SIDE_CPU;
        std::vector<std::pair<Side, ffmpeg::AVPacketPtr>> kicks_to_send;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            // 1. 跳过已供完的 GOP。⚠️ 开放 GOP（未关闭）即使瞬时供完也
            // 不能越过——后续还有包到达，越过=该 GOP 剩余包永远无人供。
            while (feed_gop_idx_ < static_cast<int64_t>(gops_.size())) {
                const GopRec &g = gops_[feed_gop_idx_];
                if (!g.closed) break;
                if (g.fed_upto < g.pkt_end) break;
                if (g.straggler_idx
                        < static_cast<int64_t>(g.stragglers.size())) {
                    break;
                }
                ++feed_gop_idx_;
            }
            // 2. 供料扫描：cursor 起、**同侧保序**的前瞻供料。
            // · 分配（供水式贪心）按扫描序 = 流序推进；
            // · 同侧乱序禁止（解码器流内序）；跨侧前瞻合法——严格序曾
            //   实测把混跑打回"交替串行"（h264 两臂各闲 26~33%）；
            // · kick（换侧 IDR 克隆）延迟化：换侧时离场侧的当前 GOP
            //   未供完 → 克隆包挂起（pending_kicks_），游标越过该 GOP
            //   再发。kick 抢先插入会打乱同侧流序（RPS 丢帧 → pending
            //   泄漏挂死，实测 "Could not find ref with POC"）。
            int64_t gi = -1;
            bool held_unassigned = false;  // 熟前挂起break置位：flush须抑制
            bool earlier_pending[2] = {false, false};
            if (feed_gop_idx_ < static_cast<int64_t>(gops_.size())) {
                const GopRec &gc = gops_[feed_gop_idx_];
                if (gc.side != Side(-1)) {
                    const int64_t ae = gc.closed ? gc.pkt_end : cache_seq_;
                    const bool partial = gc.fed_upto < ae
                        || (gc.closed && gc.straggler_idx
                                       < static_cast<int64_t>(gc.stragglers.size()));
                    if (partial) earlier_pending[(int)gc.side] = true;
                }
            }
            for (int64_t k = feed_gop_idx_;
                 k < static_cast<int64_t>(gops_.size()); ++k) {
                GopRec &g = gops_[k];
                const int64_t avail_end = g.closed ? g.pkt_end : cache_seq_;
                const bool has_pkt = g.fed_upto < avail_end
                    || (g.closed && g.straggler_idx
                                    < static_cast<int64_t>(g.stragglers.size()));
                if (k > feed_gop_idx_ && g.side != Side(-1)
                        && earlier_pending[(int)g.side]) {
                    continue;   // 同侧序约束（只约束游标之后的 GOP——
                                // 游标自身是最早，永不被自己的阻断跳过）
                }
                if (g.side == Side(-1)) {
                    if (!has_pkt) continue;   // 未开 GOP 无包不分配
                    // 硬窗界：assigned 达窗后不再派新 GOP（扫描终止；
                    // 已派 GOP 的供给与 kick 注入不受影响）。EOF 排空
                    // 同走此门，路径一致。
                    if (window_frames_ > 0
                            && assigned_frames_[0] + assigned_frames_[1]
                                   >= window_frames_) {
                        break;
                    }
                    // 熟前挂起（2026-09-18 启动轮）：rc/rg 任一未出版时
                    // 分配挂起、GOP 留在包缓存等速率成熟——此前 demux 远
                    // 快于解码，w3000 hevc 的 13 个 GOP 有 10~13 个在速率
                    // 成熟前被 50/50 交替盲派（CPU 臂超分 200+ 帧 = 慢臂
                    // 尾部 0.24s+；_probe_hybrid_startup 实测）。挂起零
                    // 成本：包缓存即为此设计。防饿死例外：①某侧空转
                    // （side_pending_==0）→ 盲派给空侧——纯库存读数，
                    // 无速率假设；②头部强制派工（force_head，Pop 饿死
                    // 解锁）：库存残帧互锁时由消费线程经 Pop 重驱泵并
                    // 强制分配头部 GOP。双空走 PickFeedSide（gop0 GPU
                    // 起步 / gop1 CPU 采样语义不变，FORCE_SIDE 亦经它
                    // 生效）。
                    const bool force_this =
                        force_head && k == feed_gop_idx_;
                    if (force_this) {
                        const bool ci = side_pending_[SIDE_CPU] == 0;
                        const bool gi = side_pending_[SIDE_GPU] == 0;
                        g.side = (ci && gi) ? PickFeedSide()
                                               : (ci ? SIDE_CPU : SIDE_GPU);
                    } else if (cpu_.ProductionRate() <= 0.0
                            || gpu_rate_landed_.load(
                                   std::memory_order_relaxed) <= 0.0) {
                        const bool cpu_idle =
                            side_pending_[SIDE_CPU] == 0;
                        const bool gpu_idle =
                            side_pending_[SIDE_GPU] == 0;
                        if (!cpu_idle && !gpu_idle) {
                            held_unassigned = true;
                            break;
                        }
                        if (ForcedSide() != Side(-1)
                                || (cpu_idle && gpu_idle)) {
                            g.side = PickFeedSide();
                        } else {
                            g.side = cpu_idle ? SIDE_CPU : SIDE_GPU;
                        }
                    } else {
                        g.side = PickFeedSide();
                    }
                    for (Chunk &c : emit_queue_) {
                        if (c.id == g.id) {
                            c.side = g.side;
                            if (c.expected > 0 && !g.counted) {
                                assigned_frames_[g.side] += c.expected;
                                g.counted = true;
                            }
                            break;
                        }
                    }
                    ++chunks_assigned_[g.side];
                    if (g.side != last_fed_side_ && last_fed_side_ != Side(-1)
                            && IsIdrLikeCodec()) {
                        // 换侧 kick：克隆本 GOP 首包（此刻必在缓存，未供）
                        // ⚠️ kick 一律挂起、由泵在流序位置注入（分配时
                        // 即发/游标触发都会在前瞻供料下落到离场侧队列尾
                        // = IDR 重置错位 → cuvid 冲掉被打断 GOP 的在途
                        // 重排窗（引擎时序下 ~50 帧遮蔽损坏 → 段数漂移
                        // 8340→8346/8353/8371；fork 均匀消费不触发故
                        // 逐帧 hash 全绿——"解码器完美但段数漂移"的谜底）。
                        ffmpeg::AVPacketPtr kp = CloneCachePacket(g.pkt_begin);
                        if (kp && pending_kicks_.size() < 8) {
                            pending_kicks_.push_back(PendingKick{
                                std::move(kp), last_fed_side_, k - 1});
                        }
                    }
                    last_fed_side_ = g.side;
                if (earlier_pending[(int)g.side]) continue;
                }
                if (!has_pkt) continue;
                // 容量门（2026-09-28 av1 死锁轮修订）：EOF 排空原本完全
                // 绕过此门（「无视饥饿门排空全部剩余包」）。但整文件进
                // 包缓存时 eof_cache_ 可在解码启动前就位——若某臂速率
                // 从未成熟（rg/rc<=0 = 零产出证据，CU 启动冻结即此形），
                // 盲排空会把全文件（实测 23647 包）倾倒给死臂：头块永等
                // + side_pending 虚高 23670 + 重试预算 FATAL。修订：EOF
                // 只对**速率已成熟**的臂保留无界排空；未熟臂照常限容
                // （消费推进持续驱动泵，Pop 的 PumpFeed(true) 保证供料
                // 不中断）。速率成熟的健康快解路径（hevc/h264 stream
                // 相位）行为不变。
                {
                    const bool side_unmatured = (g.side == SIDE_CPU)
                        ? cpu_.ProductionRate() <= 0.0
                        : gpu_rate_landed_.load(
                              std::memory_order_relaxed) <= 0.0;
                    if (!eof_cache_ || side_unmatured) {
                        const int64_t cap = (g.side == SIDE_CPU)
                            ? static_cast<int64_t>(queue_frames_)
                            : static_cast<int64_t>(ready_cap_frames_);
                        if (side_pending_[g.side] >= cap) {
                            earlier_pending[(int)g.side] = true;
                            continue;
                        }
                    }
                }
                gi = k;
                break;
            }
            if (gi < 0) {
                // 3. 无可供工作：EOF 或窗界关断、且全部已派工作供完
                // （含 kick 注入完毕）→ 置排空标记（锁外发送）。
                // 窗界即调度语义上的流结束：消费者声明只取 window 帧，
                // 已派 GOP 供完 = 窗内工作全部完成，此后臂应收尾退出——
                // 否则 worker 等不到排空标记，close 时 join 死锁
                // （2026-09-17 硬窗界首轮实测挂死，即漏了此分支）。
                const bool window_closed = window_frames_ > 0
                    && assigned_frames_[0] + assigned_frames_[1]
                           >= window_frames_;
                // 排空前确认：所有已派 GOP 已喂完（主区间+迟到包）。
                // 被侧容量门挡住的 GOP 让 gi<0 但喂包未完——此刻发
                // flush = 已派未喂帧丢失（2026-09-19 test6_h264 CPU-out
                // 窗口实测：消费者等不到窗内帧 → EOF 兜底长自旋）。
                // 容量门由消费推进自然解除（头部 chunk 即被消费），
                // 下一轮泵送继续喂完后再排空。
                bool all_assigned_fed = true;
                for (const GopRec &g3 : gops_) {
                    if (g3.side == Side(-1)) continue;
                    const int64_t ae3 = g3.closed ? g3.pkt_end : cache_seq_;
                    if (g3.fed_upto < ae3
                            || (g3.closed && g3.straggler_idx
                                           < static_cast<int64_t>(
                                               g3.stragglers.size()))) {
                        all_assigned_fed = false;
                        break;
                    }
                }
                // 窗界关断不必等游标到表尾（窗外 GOP 永不派，游标
                // 停在窗缘）——NVDEC 的 DPB 扣留尾帧必须靠 flush 释放，
                // 否则消费者等不到窗口末帧（首轮实测挂死即此）。
                if ((eof_cache_ || window_closed) && all_assigned_fed
                        && !held_unassigned && !arm_flush_sent_) {
                    arm_flush_sent_ = true;
                    eof_pushed_ = true;   // Pop 的 EOF 语义在此就位
                    if (window_closed && !eof_cache_) {
                        // 窗界排空：仅丢弃触发点已越过窗缘的 kick（其
                        // 目标 GOP 未派=窗外，注入=解码窗外帧）。目标
                        // GOP 已派（窗内）的 kick 必须保留并注入——否则
                        // 该侧 chunk 缺 IDR 重锚、expected 补不满 →
                        // 少交付（2026-09-19 test6_h264 w12000 实测
                        // EOF-retry 上限报错的根因）。
                        int64_t last_assigned_gop = -1;
                        for (auto rit = gops_.rbegin();
                             rit != gops_.rend(); ++rit) {
                            if (rit->side != Side(-1)) {
                                last_assigned_gop = rit->id;
                                break;
                            }
                        }
                        for (auto it = pending_kicks_.begin();
                             it != pending_kicks_.end();) {
                            if (it->after_gop >= last_assigned_gop) {
                                it = pending_kicks_.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    }
                }
                break;
            }
            GopRec &g = gops_[gi];
            s = g.side;
            // 取一个包（主区间优先，随后迟到包）
            const int64_t avail_end = g.closed ? g.pkt_end : cache_seq_;
            if (g.fed_upto < avail_end) {
                const int64_t seq = g.fed_upto;
                pkt = std::move(pkt_cache_[seq - cache_base_seq_]);
                cache_bytes_ -= static_cast<size_t>(pkt->size);
                is_key = (pkt->flags & AV_PKT_FLAG_KEY) != 0;
                ++g.fed_upto;
            } else {
                pkt = std::move(g.stragglers[g.straggler_idx++]);
                is_key = false;
            }
            // kick 泵注入：本包是 dst 侧在 after_gop 之后的首个被供
            // GOP 主区间首包 → kick 先发（记账锁内、推包锁外）。位置 =
            // K-1 末包之后、后续 GOP 之前——与 main 的流内 kick 等价。
            if (g.fed_upto == g.pkt_begin + 1) {   // 刚取的是主区间首包
                static const bool koff = [] {
                    const char *e = getenv("DECORD_HYBRID_KICK_OFF");
                    return e != nullptr && atoi(e) > 0;
                }();
                if (!koff) {
                    for (auto it = pending_kicks_.begin();
                         it != pending_kicks_.end(); ) {
                        if (it->dst == s && g.id > it->after_gop) {
                            ++side_pending_[s];
                            kicks_[s].fetch_add(1, std::memory_order_relaxed);
                            kicks_to_send.emplace_back(s, std::move(it->pkt));
                            it = pending_kicks_.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }
            ++side_pending_[s];
            // 2026-09-19 冻结前清理：此处原有 arm_kick 恒假分支（其唯一
            // 赋值点在 fef3c4b 重设计时删除），连带 kick_dst 死变量。
            // 现直接走"已武装则按债务克隆"的活分支。
            if (kick_side_ != Side(-1)) {
                if (side_pending_[kick_side_] - kick_cloned_ <= 0) {
                    kick_side_ = Side(-1);
                    kick_owed_ = kick_cloned_ = 0;
                } else if (is_key) {
                    kick_side_ = Side(-1);
                    kick_owed_ = kick_cloned_ = 0;
                } else if (s != kick_side_ && kick_cloned_ < kick_guard_) {
                    ++side_pending_[kick_side_];
                    ++kick_cloned_;
                    fb_clones_.fetch_add(1, std::memory_order_relaxed);
                    clone_debt = true;
                    debt_dst = kick_side_;
                }
            }
        }  // ~lock
        if (!pkt) break;   // 防御（不应发生）
        static const bool kick_off = [] {
            const char *e = getenv("DECORD_HYBRID_KICK_OFF");
            return e != nullptr && atoi(e) > 0;
        }();
        for (auto &ks : kicks_to_send) {
            if (ks.first == SIDE_CPU) {
                cpu_.Push(std::move(ks.second), runtime::NDArray());
            } else {
                std::lock_guard<std::mutex> lk(lcv_mtx_);
                gpu_pkt_q_.push_back(std::move(ks.second));
                lcv_.notify_all();
            }
        }
        if (clone_debt) {
            AVPacket *dup = av_packet_clone(pkt.get());
            CHECK(dup != nullptr) << "av_packet_clone failed";
            ffmpeg::AVPacketPtr dup_ptr(dup, [](AVPacket *p) { av_packet_free(&p); });
            if (debt_dst == SIDE_CPU) {
                cpu_.Push(std::move(dup_ptr), runtime::NDArray());
            } else {
                std::lock_guard<std::mutex> lk(lcv_mtx_);
                gpu_pkt_q_.push_back(std::move(dup_ptr));
                lcv_.notify_all();
            }
        }
        if (s == SIDE_CPU) {
            cpu_.Push(std::move(pkt), runtime::NDArray());
        } else {
            {
                std::lock_guard<std::mutex> lk(lcv_mtx_);
                gpu_pkt_q_.push_back(std::move(pkt));
            }
            lcv_.notify_all();
        }
        // 缓存回收：弹出已消费前缀（取包时已扣字节，这里只收槽位）
        {
            std::lock_guard<std::mutex> lk(mtx_);
            int64_t front = INT64_MAX;
            for (int64_t k = feed_gop_idx_;
                 k < static_cast<int64_t>(gops_.size()); ++k) {
                const GopRec &g2 = gops_[k];
                const int64_t avail_end2 = g2.closed ? g2.pkt_end : cache_seq_;
                if (g2.fed_upto < avail_end2) {
                    front = g2.fed_upto;
                    break;
                }
            }
            while (!pkt_cache_.empty() && cache_base_seq_ < front) {
                pkt_cache_.pop_front();
                ++cache_base_seq_;
            }
        }
    }
    // EOF 排空标记（锁外发送，一次）
    bool send_flush = false;
    std::vector<std::pair<Side, ffmpeg::AVPacketPtr>> eof_kicks;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        send_flush = arm_flush_sent_ && !eof_flush_out_;
        if (send_flush) {
            eof_flush_out_ = true;
            // EOF 兜底：未注入的 kick 全发（IDR 重置冲刷离场侧尾帧）
            for (auto it = pending_kicks_.begin();
                 it != pending_kicks_.end(); ) {
                ++side_pending_[it->dst];
                kicks_[it->dst].fetch_add(1, std::memory_order_relaxed);
                eof_kicks.emplace_back(it->dst, std::move(it->pkt));
                it = pending_kicks_.erase(it);
            }
        }
    }
    for (auto &ks : eof_kicks) {
        if (ks.first == SIDE_CPU) {
            cpu_.Push(std::move(ks.second), runtime::NDArray());
        } else {
            std::lock_guard<std::mutex> lk(lcv_mtx_);
            gpu_pkt_q_.push_back(std::move(ks.second));
            lcv_.notify_all();
        }
    }
    if (send_flush) {
        cpu_.Push(nullptr, runtime::NDArray());
        if (gpu_) {
            {
                std::lock_guard<std::mutex> lk(lcv_mtx_);
                gpu_flush_left_ = ThreadedDecoderInterface::kMaxOutputSurfaces;
            }
            lcv_.notify_all();
        }
    }
}


void HybridThreadedDecoder::Push(ffmpeg::AVPacketPtr pkt, runtime::NDArray buf) {
    // 包缓存版（层0/1）：Push 只入缓存 + GOP 记账，供料泵在分配时点
    // 派工（见 PumpFeed）。VideoReader 对 hybrid 一律推 NDArray()（与
    // CPU 分支一致）；GPU 侧输出缓冲由工作线程从有界池 Acquire。
    if (!pkt) {
        {   // EOF：关闭当前 GOP，泵排空后由 PumpFeed 发臂排空标记
            std::lock_guard<std::mutex> lk(mtx_);
            if (routing_active_ && !gops_.empty() && !gops_.back().closed) {
                CloseGopLocked();
            }
            eof_cache_ = true;
        }
        PumpFeed();
        return;
    }
    const bool is_key = (pkt->flags & AV_PKT_FLAG_KEY) != 0;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!routing_active_) {
            routing_active_ = true;
            stats_t0_us_.store(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count(),
                std::memory_order_relaxed);
        }
        if (is_key) {
            if (!gops_.empty() && !gops_.back().closed) {
                if (!emit_queue_.empty()) {
                    emit_queue_.back().end_pts = pkt->pts;  // 先写 end
                }
                CloseGopLocked();
            }
            OpenGopLocked(pkt->pts);
        }
        // 迟到包（重排：非 key 且 pts 早于当前 GOP 起点）归属更早的 GOP：
        // 已供 → 直供其侧（旧行为）；未供 → 挂 stragglers 随主区间供。
        // 非 IDR 型（AV1）无法归属的包跟随当前 GOP（旧语义：孤立中段包
        // 喂软解会 dav1d parse 崩溃，其输出旧帧由陈旧丢弃兜底）。
        bool stashed = false;
        if (!is_key && gops_.size() >= 2) {
            const int64_t cur_start = gops_.back().start_pts;
            if (pkt->pts < cur_start) {
                GopRec *own = nullptr;
                if (IsIdrLikeCodec()) {
                    for (auto it = gops_.rbegin() + 1;
                         it != gops_.rend(); ++it) {
                        if (pkt->pts >= it->start_pts) { own = &*it; break; }
                    }
                } else {
                    own = &gops_[gops_.size() - 2];   // 前一 GOP
                }
                if (own) {
                    if (own->fed_upto >= (own->closed ? own->pkt_end
                                                      : cache_seq_)) {
                        late_feeds_.fetch_add(1, std::memory_order_relaxed);
                        const Side s2 = own->side;
                        if (s2 != Side(-1)) {
                            ++side_pending_[s2];
                            if (s2 == SIDE_CPU) {
                                cpu_.Push(std::move(pkt), runtime::NDArray());
                            } else {
                                std::lock_guard<std::mutex> lk2(lcv_mtx_);
                                gpu_pkt_q_.push_back(std::move(pkt));
                            }
                            stashed = true;
                        }
                    } else {
                        own->stragglers.push_back(std::move(pkt));
                        strag_total_.fetch_add(1, std::memory_order_relaxed);
                        stashed = true;
                    }
                }
            }
        }
        if (!stashed) {
            cache_bytes_ += static_cast<size_t>(pkt->size);
            if (cache_bytes_ > cache_peak_bytes_) {
                cache_peak_bytes_ = cache_bytes_;
            }
            pkt_cache_.push_back(std::move(pkt));
            ++cache_seq_;
        }
    }
    PumpFeed();
}


void HybridThreadedDecoder::OpenGopLocked(int64_t pts) {
    // 打开 GOP：交付 chunk（side 待分配）+ 缓存记账记录。仅 Push 线程
    //（持 mtx_）调用。
    GopRec g;
    g.id = gop_seq_;
    g.pkt_begin = cache_seq_;
    g.pkt_end = cache_seq_;
    g.fed_upto = cache_seq_;   // 绝对序列语义：初始=区间起点
    g.start_pts = pts;
    gops_.push_back(std::move(g));
    Chunk c{Side(-1), gop_seq_, pts, INT64_MAX, 0, 0};
    emit_queue_.push_back(c);
    ++gop_seq_;
}


void HybridThreadedDecoder::CloseGopLocked() {
    // 关闭当前 GOP（end_pts 已由调用方写入 chunk）。expected 从 kf 索引
    // 取；分配时累计进 assigned_frames_（counted 防双计）。
    if (gops_.empty() || emit_queue_.empty()) return;
    GopRec &g = gops_.back();
    const Chunk &last = emit_queue_.back();
    const int64_t end_pts = last.end_pts;
    g.closed = true;
    g.pkt_end = cache_seq_;
    int64_t exp = ExpectedFrames(last.start_pts, end_pts);
    if (end_pts == INT64_MAX && exp <= 0) {
        exp = est_chunk_frames_ > 0 ? est_chunk_frames_ : 0;
    }
    if (exp > 0) est_chunk_frames_ = exp;
    for (auto it = emit_queue_.rbegin(); it != emit_queue_.rend(); ++it) {
        if (it->id == g.id) { it->expected = exp; it->end_pts = end_pts; break; }
    }
    if (exp > 0 && g.side != Side(-1) && !g.counted) {
        assigned_frames_[g.side] += exp;
        g.counted = true;
    }
}


bool HybridThreadedDecoder::IsMarker(const runtime::NDArray &f) {
    if (!f.defined()) return false;
    const DLTensor *t = f.operator->();
    return t != nullptr && t->dtype.code == kDLInt && t->dtype.bits == 64;
}


bool HybridThreadedDecoder::PopSide(Side s, runtime::NDArray *f) {
    // 取帧优先级：stash（越界暂存）→ 子解码器 / ready_ 队列
    if (has_stash_[s]) {
        *f = stash_[s];
        has_stash_[s] = false;
        return true;
    }
    if (s == SIDE_CPU) {
        if (out_cuda_) {
            // GPU 驻留模式：取已上载的显存帧（marker 由 UploadStep 直通），非阻塞
            std::lock_guard<std::mutex> lk(rmtx_);
            if (cpu_ready_.empty()) return false;
            *f = std::move(cpu_ready_.front());
            cpu_ready_.pop_front();
            lcv_.notify_all();  // 槽位/预算释放：即时唤醒喂包/上载线程
            return f->defined();
        }
        // 非阻塞门（2026-09-27 停滞修复）：cpu_.Pop 是阻塞队列操作——
        // EOF 尾会计失配（marker 被「CPU 侧未推流」分支误食后 chunk 仍
        // 期望帧）时会**永眠**并绕过 stall 看门狗与 VideoReader 的 EOF
        // 重试两层防线（av1 无 kick 路径实测卡死）。本类是 cpu_ 的唯一
        // 消费者：QueueDepth()>0 ⇒ Pop 必然立即返回（生产者只入队，
        // 无 TOCTOU）；深度 0 ⇒ 返回 false 交上层重试/安全网处置——
        // 这也是头文件「消费者线程的 Pop 永不阻塞」设计承诺的落位
        //（CPU 侧此前是唯一违例）。
        if (cpu_.QueueDepth() == 0) return false;
        return cpu_.Pop(f);
    }
    // GPU 帧：取工作线程已落地/直通的帧，非阻塞
    std::lock_guard<std::mutex> lk(rmtx_);
    if (ready_.empty()) return false;
    *f = std::move(ready_.front());
    ready_.pop_front();
    lcv_.notify_all();  // 槽位/预算释放：即时唤醒喂包线程
    return f->defined();
}


bool HybridThreadedDecoder::Pop(runtime::NDArray *frame) {
    // ── HOL 时长累账（非打印，见 .h 注释）：上次调用停在队头阻塞、本
    // 调用入口之间的墙钟计入该侧；episode 保持开启直到发射成功或双侧
    // 全空。单消费者线程读写，无并发问题。──
    if (hol_prev_side_ >= 0) {
        const auto now_tp = std::chrono::steady_clock::now();
        int64_t dt = std::chrono::duration_cast<std::chrono::microseconds>(
                         now_tp - hol_tp_).count();
        if (dt > 50000) dt = 50000;  // 夹掉消费间隙（GC/probe 收尾等）
        if (dt > 0) {
            hol_us_[hol_prev_side_].fetch_add(
                dt, std::memory_order_relaxed);
        }
        hol_tp_ = now_tp;
    }
        if (err_seen_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lk(err_mtx_);
            throw dmlc::Error("hybrid worker failed: " + err_msg_);
        }
    while (true) {
        Chunk ch;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (emit_queue_.empty()) {
                break;
            }
            ch = emit_queue_.front();
        }
        // 未分配的队头（供料泵尚未派工——只在启动瞬态出现）：等同"暂无
        // 帧可取"，返回 false 等泵供上（force-close 安全网也依赖 side
        // 合法，先挡掉）。
        if (ch.side == Side(-1)) {
            if (eof_pushed_) {
                // 排空已启动后仍未派工的 chunk = 窗外 GOP（无帧可失）
                // ——移除以让排空 marker 可达，否则 marker 被
                // return false 挡住 → 消费者静默自旋（pop-stall 取证
                // 只覆盖已派 chunk，覆盖不到这里）。
                std::lock_guard<std::mutex> lk(mtx_);
                if (!emit_queue_.empty()
                        && emit_queue_.front().start_pts == ch.start_pts
                        && emit_queue_.front().side == Side(-1)) {
                    emit_queue_.pop_front();
                    continue;
                }
            }
            // 取证硬化（2026-09-28 夜间轮）：av1 hybrid 流中段互锁卡死
            // 的三条静默路径之一——此分支无任何诊断，FATAL 后零现场。
            // 与 PopSide 失败分支同款时间门控（3s 首报/指数退避），仅
            // 失败路径触达，健康路径零成本。
            {
                const auto now_ = std::chrono::steady_clock::now();
                bool due = std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                        now_ - stall_last_tp_).count() >= 3000;
                if (due && stall_delay_ms_ == 0) {
                    stall_print_tp_ = now_;
                    stall_delay_ms_ = 1000;
                    std::size_t eqn = 0, gsz = 0;
                    {
                        std::lock_guard<std::mutex> lk(mtx_);
                        eqn = emit_queue_.size();
                        gsz = gops_.size();
                    }
                    fprintf(stderr,
                            "\n[pop-stall3] unassigned-head eq=%zu gops=%zu "
                            "pend=%d/%d infl=%lld fp=%d\n",
                            eqn, gsz, (int)side_pending_[0],
                            (int)side_pending_[1],
                            (long long)(side_pending_[0] + side_pending_[1]),
                            (int)eof_pushed_);
                    fflush(stderr);
                }
            }
            // 熟前挂起配套：EOF/窗界后 Push 停驱，未分配 GOP 由消费
            // 线程在此重驱泵（分配可能此刻已可进行——速率成熟/防饿死）。
            // force=true：头部已饿死仍不能分配 = 库存残帧互锁（两侧
            // pending 均非零但都不属于头部可发射序）——强制派工头部
            // GOP 解锁。仅此处使用；正常推进走 Push 路径的无参泵。
            PumpFeed(true);
            return false;
        }
        // force-close 安全网：该侧已路由的包全部出清（pending==0）但
        // expected 未补满 —— 重排流的跨侧切换残留（无 kick 的 AV1 混合
        // 路由）。接受缺失帧关闭 chunk（后续陈旧丢弃保序），避免无限等待。
        if (!eof_pushed_ && ch.expected > 0 && ch.emitted < ch.expected
                && side_pending_[ch.side] == 0) {
            bool close_it = false;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (emit_queue_.size() > 1
                        && emit_queue_.front().start_pts == ch.start_pts
                        && emit_queue_.front().emitted == ch.emitted
                        && side_pending_[ch.side] == 0) {
                    emit_queue_.pop_front();
                    close_it = true;
                }
            }
            if (close_it) {
                if (getenv("DECORD_HYBRID_DEBUG")) {
                    fprintf(stderr, "[hybrid] force-close chunk start=%lld emitted=%lld/%lld\n",
                            (long long)ch.start_pts, (long long)ch.emitted,
                            (long long)ch.expected);
                }
                continue;
            }
        }
        Side s = ch.side;
        runtime::NDArray f;
        if (!PopSide(s, &f)) {
            static const bool dbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
            // 取证采样**默认开**（2026-09-12 §18 硬化：挂死现场自带签名，
            // 无需先复现再开 STATS）。判别器是**时间**不是计数——正常供帧
            // 间隙（毫秒级）就能凑满任意连败计数（实测健康路径 2000 连败
            // 每秒触发多次）；无进展 ≥3s 才首报，之后 1s 起指数退避至 60s
            // 上限。挂死 ≈ 3s 出首条、首分钟 ~10 条、之后 1 条/分钟；健康
            // 路径供帧间隙远小于 3s，零误报。steady_clock ~20ns/次且只在
            // 失败路径（本就走锁），扰动可忽略。
            {
                const auto now_ = std::chrono::steady_clock::now();
                bool due = false;
                if (stall_delay_ms_ == 0) {
                    due = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now_ - stall_last_tp_).count() >= 3000;
                } else {
                    due = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now_ - stall_print_tp_).count() >= stall_delay_ms_;
                }
                if (due) {
                    stall_print_tp_ = now_;
                    stall_delay_ms_ = stall_delay_ms_
                        ? std::min<int64_t>(stall_delay_ms_ * 2, 60000)
                        : 1000;
                    // 连续取空诊断：头部 chunk 与两侧队列状态（D 类定位用）。
                    // 只嵌 rmtx_ 读队列长度；chunk/pend 字段按本文件既有
                    // 调试打印惯例无锁读取（仅诊断，不保证精确快照）。
                    {
                    std::size_t crdy, rdy;
                    {
                        std::lock_guard<std::mutex> lk(rmtx_);
                        crdy = cpu_ready_.size();
                        rdy = ready_.size();
                    }
                    std::size_t qpk = 0;
                    {
                        std::lock_guard<std::mutex> lk2(lcv_mtx_);
                        qpk = gpu_pkt_q_.size();
                    }
                    std::size_t eqn = 0; int stm = 0;
                    {
                        std::lock_guard<std::mutex> lk3(mtx_);
                        eqn = emit_queue_.size();
                        stm = (has_stash_[0] ? 1 : 0) | (has_stash_[1] ? 2 : 0);
                    }
                    int64_t cud_p = -1, cud_b = -1, cud_o = -1;
                    if (gpu_) gpu_->DiagDepths(&cud_p, &cud_b, &cud_o);
                    fprintf(stderr,
                            "\n[pop-stall] side=%d crdy=%zu rdy=%zu "
                            "head=(side%d,%lld,end=%lld,exp=%lld,em=%lld) "
                            "pend=%d/%d\n"
                            "[pop-stall2] cq=%lld cp=%lld qpk=%zu eq=%zu stm=%d "
                            "gpf=%lld gpc=%lld upf=%lld upc=%lld "
                            "np=%d si=%d pr=%d kick=%lld/%lld fb=%d:%d/%d"
                            " infl=%lld cud=%lld/%lld/%lld\n",
                            (int)s, crdy, rdy,
                            (int)ch.side, (long long)ch.start_pts,
                            (long long)ch.end_pts, (long long)ch.expected,
                            (long long)ch.emitted, (int)side_pending_[0],
                            (int)side_pending_[1],
                            (long long)cpu_.QueueDepth(),
                            (long long)cpu_.PendingDepth(),
                            qpk, eqn, stm,
                            (long long)gpu_pool_.DiagFree(),
                            (long long)gpu_pool_.DiagCreated(),
                            (long long)up_pool_.DiagFree(),
                            (long long)up_pool_.DiagCreated(),
                            (int)NeedsPackets(), (int)sched_initialized_,
                            (int)eof_cache_,
                            (long long)kicks_[0].load(),
                            (long long)kicks_[1].load(),
                            (int)kick_side_, kick_cloned_, kick_owed_,
                            (long long)(side_pending_[0] + side_pending_[1]),
                            cud_p, cud_b, cud_o);
                    }
                }
            }
            if (dbg) fprintf(stderr, "[hybrid-p] empty side=%d emitted_total=%lld\n", (int)s, (long long)emitted_total_);
            {   // HOL 判定：head 侧空但**对侧有存货** = 保序队头阻塞（对侧
                // 帧已产出却被排在后面的 chunk 卡住）；双侧全空 = 真·生产
                // 不足，不计 HOL。
                int64_t other = 0;
                if (s == SIDE_GPU) {
                    if (out_cuda_) {
                        std::lock_guard<std::mutex> lk(rmtx_);
                        other = static_cast<int64_t>(cpu_ready_.size());
                    } else {
                        other = static_cast<int64_t>(cpu_.QueueDepth());
                    }
                } else {
                    std::lock_guard<std::mutex> lk(rmtx_);
                    other = static_cast<int64_t>(ready_.size());
                }
                if (other > 0) {
                    if (hol_prev_side_ < 0) {
                        hol_ev_[s].fetch_add(1, std::memory_order_relaxed);
                        // 穿透轮：episode 入口的对侧存货分布（log2 桶×24）
                        int hb = 0;
                        for (int64_t v = other; v > 1; v >>= 1) ++hb;
                        hol_hist_[s][hb < 23 ? hb : 23].fetch_add(
                            1, std::memory_order_relaxed);
                        hol_prev_side_ = static_cast<int>(s);
                        hol_tp_ = std::chrono::steady_clock::now();
                    }
                    int64_t mx = hol_strand_max_[s].load(std::memory_order_relaxed);
                    while (other > mx
                           && !hol_strand_max_[s].compare_exchange_weak(
                               mx, other, std::memory_order_relaxed)) {}
                } else {
                    hol_prev_side_ = -1;
                }
            }
            // ── EOF 尾恢复网（2026-09-27 停滞修复；2026-09-28 对称扩展 v2）──
            // 触发面：eof_pushed_ 已置 + 队头 chunk 所在侧持续 ≥1s 零产出。
            // 根因是 marker 顺序竞态：侧排空 marker 在 eof_pushed_ 置位前
            // 到达时走「吞掉」分支被误食，真到 EOF 时无 marker 可收口，
            // chunk 会计（expected/pending）与解码器实际产出永久失配。
            // 时间基准（非计数）：大 GOP 解码突发 >100ms 属正常，计数判别
            // 会误伤；1s 空产出在 EOF 后只可能是失配。
            //
            // 真·尾部判别（对称扩展 v2，2026-09-28 发布轮）：首轮裸对称
            // 扩展（直接去掉 s==SIDE_CPU）已回退——av1 整文件装进 512MB
            // 包缓存时 demux 秒完，eof_pushed_ 在**消费中段**即置位，而
            // GPU 臂池互锁的合法停滞 ≥3s（formats 运行捕获）会被 1s 网
            // 误关有帧 chunk → 帧永久丢失 → stream 套件 3/5 FATAL。v2
            // 对 GPU 侧增加 side_pending_[s]==0 门：该侧仍有在途工作
            // （已解码未上载/未消费的帧都计入 pending，合法互锁停滞必然
            // pending>0）时永不强关，只在「该侧确无任何可再到达的帧」时
            // 收口。CPU 半边保持首轮语义（16/16 绿证据面，不加门）：软解
            // 无池互锁类合法长停，1s 空窗本就只为失配设计。
            // 命中 → 按既有 force-close 同款语义关队头 chunk（身份复查
            // 防并发漂移），计数器入 stats。marker 正常到达的路径不受
            // 影响（任何产出都会走下面的重置）。
            if (eof_pushed_ && !emit_queue_.empty()
                    && (s == SIDE_CPU || side_pending_[s] == 0)) {
                const auto now_tp = std::chrono::steady_clock::now();
                if (!eof_starve_on_) {
                    eof_starve_on_ = true;
                    eof_starve_tp_ = now_tp;
                } else if (std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        now_tp - eof_starve_tp_).count() >= 1000) {
                    bool closed = false;
                    {
                        std::lock_guard<std::mutex> lk(mtx_);
                        if (!emit_queue_.empty()
                                && emit_queue_.front().id == ch.id) {
                            emit_queue_.pop_front();
                            closed = true;
                        }
                    }
                    if (closed) {
                        force_eof_close_.fetch_add(1,
                            std::memory_order_relaxed);
                        fprintf(stderr,
                                "[hybrid] eof-starve force-close chunk id=%lld"
                                " side=%d emitted=%lld/%lld\n",
                                (long long)ch.id, (int)ch.side,
                                (long long)ch.emitted,
                                (long long)ch.expected);
                        fflush(stderr);
                        eof_starve_on_ = false;
                        continue;
                    }
                }
            }
            // ── GPU 臂冻结检测（2026-09-28 av1 死锁轮根治位）───────────
            // CU 解码臂启动竞态冻结签名：包滞留解析队列（cud_p>0）、重排
            // 环零帧（cud_o==0）、GPU 侧发射 3s 零增量。冻结时队头 GPU
            // chunk 的帧永不可达——与其耗尽重试预算静默 FATAL，不如带
            // 终态大声失败（DECORDError）。配套防线：泵侧 EOF 排空对
            // 未熟臂限容（见 PumpFeed 容量门注释），本签名触发时
            // side_pending 已被限在有界量级。健康路径不命中：慢落地时
            // ord 非零波动、fg 缓增；只有「有包不解析 + 零帧完成」才是
            // 解码臂死锁。
            if (s == SIDE_GPU && gpu_) {
                const auto now_g = std::chrono::steady_clock::now();
                const int64_t fg_now = frames_out_[1].load(
                    std::memory_order_relaxed);
                int64_t cp_ = -1, cb_ = -1, co_ = -1;
                gpu_->DiagDepths(&cp_, &cb_, &co_);
                const bool frozen_sig = cp_ > 0 && co_ == 0;
                if (frozen_sig && gpu_stall_fg_ == fg_now) {
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(
                            now_g - gpu_stall_tp_).count() >= 3000) {
                        gpu_arm_stall_.fetch_add(1,
                            std::memory_order_relaxed);
                        arm_stalled_.store(true, std::memory_order_release);
                        fprintf(stderr,
                                "[hybrid] GPU arm stalled: CU decode worker "
                                "frozen (pkt_queue=%lld unconsumed, "
                                "reorder=0, no output 3s) - signaling "
                                "VideoReader self-heal\n",
                                (long long)cp_);
                        fflush(stderr);
                    }
                } else if (frozen_sig || gpu_stall_fg_ != fg_now) {
                    gpu_stall_tp_ = now_g;
                    gpu_stall_fg_ = fg_now;
                }
            }
            PumpFeed(true);   // 同 A4：空取=头部饥饿，强制泵推进
            return false;
        }
        // PopSide 成功 = 有进展（marker/陈旧丢弃/越界暂存/发射同此之后）：
        eof_starve_on_ = false;   // EOF 尾恢复网的空窗计时清零
        // stall 报警状态在此清零（见上方 stall 取证注释）。
        stall_last_tp_ = std::chrono::steady_clock::now();
        stall_delay_ms_ = 0;
        if (IsMarker(f)) {
            // 子解码器排空（EOF 后出现）
            if (s == SIDE_CPU && eof_pushed_) {
                // 全局 EOF：清空合并队列，marker 原样转发给调用方
                //（NextFrameImpl 的 kInt64 分支依赖它走 EOF 逻辑；后续
                // marker 继续透传）
                std::lock_guard<std::mutex> lk(mtx_);
                emit_queue_.pop_front();
                emit_queue_.clear();
                hol_prev_side_ = -1;
                *frame = f;
                return true;
            }
            // GPU drain marker（或 CPU 侧未推流的 marker）：吞掉，
            // chunk 完成，继续下一 chunk
            std::lock_guard<std::mutex> lk(mtx_);
            emit_queue_.pop_front();
            continue;
        }
        if (f.pts < ch.start_pts) {
            // 陈旧帧（kick 边界帧在 expected 补齐后残留等）：丢弃
            // 取证硬化（2026-09-28 夜间轮）：陈旧排放活锁假设——每次
            // Pop 内部丢一帧（PopSide 成功复位停滞计时器）后取空返回
            // false，消费者 1ms 重试再丢一帧 → 万级积压 ≈ 数十秒「假
            // 停滞」，零 pop-stall 打印（计时器恒被复位）。消费者线程
            // 累计计数（健康流全程仅十级，4096 阈值只在病理性排放触达）。
            {
                static thread_local int64_t stale_in_call = 0;
                ++stale_in_call;
                if (stale_in_call % 4096 == 0) {
                    fprintf(stderr,
                            "\n[stale-drain] n=%lld head=(side%d,%lld,%lld) "
                            "pend=%d/%d em=%lld/%lld eq=%zu fp=%d\n",
                            (long long)stale_in_call, (int)ch.side,
                            (long long)ch.start_pts, (long long)ch.end_pts,
                            (int)side_pending_[0], (int)side_pending_[1],
                            (long long)ch.emitted, (long long)ch.expected,
                            emit_queue_.size(), (int)eof_pushed_);
                    fflush(stderr);
                }
            }
            std::lock_guard<std::mutex> lk(mtx_);
            --side_pending_[s];
            continue;
        }
        if (ch.end_pts != INT64_MAX && f.pts >= ch.end_pts) {
            // 越界帧（kick 产物 / 该侧更晚 chunk 的首帧）：stash 扣住，
            // 不关 chunk —— 本 chunk 的重排尾部帧（pts < end）可能仍在途。
            // 迟到帧继续从该侧产出、正常发射计数；expected 补满时在
            // 发射路径关闭 chunk。
            std::lock_guard<std::mutex> lk(mtx_);
            CHECK(!has_stash_[s]) << "stash overflow: side " << s;
            stash_[s] = f;
            has_stash_[s] = true;
            continue;
        }
        // 发射该帧（属于本 chunk）；expected 补满且非末 chunk 则关闭
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!emit_queue_.empty()) {
                Chunk &front = emit_queue_.front();
                ++front.emitted;
                if (front.expected > 0 && front.emitted >= front.expected
                        && front.end_pts != INT64_MAX) {
                    // 补满 expected 且非末 chunk：关闭。stash 里的越界帧
                    // 轮到该侧下一 chunk 时先出（pts 落在其区间或陈旧丢弃）
                    emit_queue_.pop_front();
                }
            }
            --side_pending_[s];
            ++emitted_total_;
        }
        frames_out_[s]++;
        {
            // 发射时间线（DECORD_HYBRID_DEBUG）：每 256 帧一条 —— 分侧
            // 累计 + 距上一条的墙钟，用于实测各阶段发射速率与交替损耗
            static const bool tdbg = getenv("DECORD_HYBRID_DEBUG") != nullptr;
            if (tdbg) {
                static thread_local int64_t tl_total = 0;
                static thread_local auto tl_tp = std::chrono::steady_clock::now();
                static thread_local int64_t tl_c = 0, tl_g = 0;
                ++tl_total;
                if (tl_total % 256 == 0) {
                    auto now = std::chrono::steady_clock::now();
                    double dt = std::chrono::duration<double>(now - tl_tp).count();
                    int64_t c = frames_out_[0].load(), g = frames_out_[1].load();
                    fprintf(stderr,
                            "\n[emit-tl] t=%.2f total=%lld c=%lld(+%lld) g=%lld(+%lld) rate=%.0f",
                            std::chrono::duration<double>(
                                now.time_since_epoch()).count(),
                            (long long)tl_total, (long long)c, (long long)(c - tl_c),
                            (long long)g, (long long)(g - tl_g),
                            dt > 0 ? 256.0 / dt : 0.0);
                    tl_tp = now; tl_c = c; tl_g = g;
                }
            }
        }
        hol_prev_side_ = -1;  // 发射成功：HOL episode 结束
        *frame = f;
        return true;
    }
    // 合并队列已空：转发 CPU 侧 drain marker，让 VideoReader 的
    // EOF/rewind 逻辑（NextFrameImpl 的 kInt64 分支）正常运转
    // （GPU 驻留模式下 PopSide(SIDE_CPU) 读已上载队列，marker 由
    // UploadStep 直通）。
    // 取证硬化（2026-09-28 夜间轮）：静默路径之三——队列空 + CPU 侧
    // 永空（最终 marker 丢失/未发）会在此永恒 false，此前零现场。
    // 同款一次性时间门控（stall_last_tp_ 由任何 PopSide 成功复位）。
    if (!PopSide(SIDE_CPU, frame)) {
        const auto now_ = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                now_ - stall_last_tp_).count() >= 3000
                && stall_delay_ms_ == 0) {
            stall_print_tp_ = now_;
            stall_delay_ms_ = 1000;
            std::size_t crdy = 0, rdy = 0;
            {
                std::lock_guard<std::mutex> lk(rmtx_);
                crdy = cpu_ready_.size();
                rdy = ready_.size();
            }
            fprintf(stderr,
                    "\n[pop-stall4] empty-queue fallthrough crdy=%zu rdy=%zu "
                    "cq=%lld cp=%lld fp=%d ef=%d/%d\n",
                    crdy, rdy, (long long)cpu_.QueueDepth(),
                    (long long)cpu_.PendingDepth(),
                    (int)eof_pushed_, (int)arm_flush_sent_,
                    (int)eof_flush_out_);
            fflush(stderr);
        }
        return false;
    }
    return true;
}


}  // namespace decord
