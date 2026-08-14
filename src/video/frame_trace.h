/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file frame_trace.h
 * \brief 帧轨迹日志（诊断 CUDA 解码竞态用，DECORD_TRACE_FRAMES=路径 时启用）
 */
#ifndef DECORD_VIDEO_FRAME_TRACE_H_
#define DECORD_VIDEO_FRAME_TRACE_H_

#include <cstdio>
#include <mutex>
#include <string>

#include <cstdlib>

namespace decord {
namespace trace {

inline FILE*& fp() {
    static FILE* f = nullptr;
    return f;
}
inline bool& init_done() {
    static bool b = false;
    return b;
}
inline std::mutex& mtx() {
    static std::mutex m;
    return m;
}

inline void ensure_init() {
    if (init_done()) return;
    init_done() = true;
    const char* p = std::getenv("DECORD_TRACE_FRAMES");
    if (p && *p) {
        fp() = std::fopen(p, "w");
    }
}

inline void log(const char* tag, long long a, long long b, long long c) {
    ensure_init();
    if (!fp()) return;
    std::lock_guard<std::mutex> lk(mtx());
    std::fprintf(fp(), "%s %lld %lld %lld\n", tag, a, b, c);
    std::fflush(fp());
}

}  // namespace trace
}  // namespace decord

#endif  // DECORD_VIDEO_FRAME_TRACE_H_
