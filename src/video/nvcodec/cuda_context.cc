/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cuda_context.cc
 * \brief CUDA Context
 */

#include "cuda_context.h"
#include "../../runtime/cuda/cuda_common.h"

namespace decord {
namespace cuda {
using namespace runtime;

CUContext::CUContext() : context_{0}, initialized_{false} {
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
}

CUContext::CUContext(CUcontext ctx)
    : context_{ctx}, initialized_{true} {
}

CUContext::~CUContext() {
    if (initialized_) {
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
    if (initialized_) {
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