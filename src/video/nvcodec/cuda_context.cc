/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_context.cc
 * \brief CUDA Context
 */

#include "cuda_context.h"
#include "../../runtime/cuda/cuda_common.h"

#include <mutex>
#include <unordered_set>

namespace decord {
namespace cuda {
using namespace runtime;

namespace {
// 进程级 keepalive:主上下文的 retain 计数一旦归零,驱动会销毁它(CUDA 文档
// 语义)——进程内所有加载进该上下文的静态句柄(improc module)与 cudart 的
// 隐式状态随之悬空,之后的新会话使用旧句柄即无声进程死亡(2026-09-08 事故,
// 强制 cuDevicePrimaryCtxReset 已复现同类损坏)。每个设备首次使用时额外
// retain 一份永不释放的引用,让主上下文活到进程退出、由驱动统一清理;
// 之后各 reader 的 retain/release 正常配对,计数地板为 1。
std::mutex g_keepalive_mu;
std::unordered_set<unsigned int> g_keepalive_devices;
}  // namespace

CUContext::CUContext() : device_{0}, context_{0}, initialized_{false} {
}

CUContext::CUContext(CUdevice device, unsigned int flags)
    : device_{device}, context_{0}, initialized_{false} {
  CHECK_CUDA_CALL(nv::cuInit(0));
  if (!CHECK_CUDA_CALL(nv::cuDevicePrimaryCtxRetain(&context_, device))) {
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed, can't go forward without a context");
  }
  Push();
  CUdevice dev;
  if (!CHECK_CUDA_CALL(nv::cuCtxGetDevice(&dev))) {
    throw std::runtime_error("Unable to get device");
  }
  initialized_ = true;
  CHECK_CUDA_CALL(nv::cuCtxSynchronize());
  {
    std::lock_guard<std::mutex> lk(g_keepalive_mu);
    if (g_keepalive_devices.insert(device).second) {
      CUcontext keepalive = nullptr;
      CHECK_CUDA_CALL(nv::cuDevicePrimaryCtxRetain(&keepalive, device));
    }
  }
}

CUContext::CUContext(CUcontext ctx)
    : device_{0}, context_{ctx}, initialized_{true} {
}

CUContext::~CUContext() {
    if (initialized_ && device_ != 0) {
        // cuCtxPopCurrent?
        CHECK_CUDA_CALL(nv::cuDevicePrimaryCtxRelease(device_));
    }
}

CUContext::CUContext(CUContext&& other)
    : device_{other.device_}, context_{other.context_},
      initialized_{other.initialized_} {
    other.device_ = 0;
    other.context_ = 0;
    other.initialized_ = false;
}

CUContext& CUContext::operator=(CUContext&& other) {
    if (initialized_ && device_ != 0) {
        // This context came from cuDevicePrimaryCtxRetain; primary contexts must be
        // released with Release (cuCtxDestroy returns CUDA_ERROR_INVALID_CONTEXT 201)
        CHECK_CUDA_CALL(nv::cuDevicePrimaryCtxRelease(device_));
    }
    device_ = other.device_;
    context_ = other.context_;
    initialized_ = other.initialized_;
    other.device_ = 0;
    other.context_ = 0;
    other.initialized_ = false;
    return *this;
}

void CUContext::Push() const {
    CUcontext current;
    if (!CHECK_CUDA_CALL(nv::cuCtxGetCurrent(&current))) {
        throw std::runtime_error("Unable to get current context");
    }
    if (current != context_) {
        // Use cuCtxSetCurrent instead of cuCtxPushCurrent:
        // after cudart init (cudaSetDevice) the primary current state is owned by cudart,
        // so cuCtxPushCurrent(primary) returns CUDA_ERROR_INVALID_CONTEXT (201);
        // SetCurrent sets it directly without a stack, per-thread.
        if (!CHECK_CUDA_CALL(nv::cuCtxSetCurrent(context_))) {
            throw std::runtime_error("Unable to set current context");
        }
    }
}

bool CUContext::Initialized() const {
    return initialized_;
}

CUContext::operator CUcontext() const {
    return context_;
}

}  // namespace cuda
}  // namespace decord