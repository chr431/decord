/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cudart_shim.cc
 * \brief CUDA Runtime API 垫片实现：全部走 nv:: 动态加载的驱动 API。
 *  驱动缺失时包装器返回 CUDA_ERROR_NOT_FOUND → 上层回退 CPU。
 */
#include "cudart_shim.h"

#include <cstdio>
#include <cstring>
#include <mutex>

#include "../../video/nvcodec/nv_gpu_dyn.h"

namespace {

/* current device → retained primary context（与 cuda_context.cc 的
 * primary-ctx 模型一致；runtime 的 per-thread current device 语义由
 * 驱动的 per-thread current context 提供）。 */
std::mutex g_ctx_mtx;
CUcontext g_ctx[64] = {};   // device ordinal → primary ctx
CUdevice g_dev[64] = {};    // device ordinal → CUdevice
bool g_ctx_tried[64] = {};  // 防重复 retain 尝试

thread_local int tls_current_device = 0;  // cudaSetDevice 的 runtime 语义

CUcontext CtxFor(int device) {
  std::lock_guard<std::mutex> lk(g_ctx_mtx);
  if (getenv("DECORD_CUDART_DEBUG")) {
    fprintf(stderr, "[cudart-shim] CtxFor(%d) tried=%d ctx=%p",
            device, (int)g_ctx_tried[device], (void *)g_ctx[device]);
  }
  if (device < 0 || device >= 64) return nullptr;
  if (!g_ctx_tried[device]) {
    CUdevice dev = -1;
    if (nv::cuDeviceGet(&dev, device) == CUDA_SUCCESS) {
      CUcontext ctx = nullptr;
      if (nv::cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS && ctx) {
        g_ctx[device] = ctx;
        g_dev[device] = dev;
        g_ctx_tried[device] = true;  // 仅成功时缓存：失败可重试（如驱动晚初始化）
      }
    }
  }
  return g_ctx[device];
}

/* runtime 语义：分配类调用作用于"当前线程的当前设备"。若当前线程尚无
 * context（从未 cudaSetDevice），等价于设备 0。 */
CUcontext EnsureCurrentCtx() {
  CUcontext cur = nullptr;
  nv::cuCtxGetCurrent(&cur);
  if (getenv("DECORD_CUDART_DEBUG")) {
    fprintf(stderr, "[cudart-shim] EnsureCurrentCtx cur=%p", (void *)cur);
  }
  if (cur != nullptr) {
    return cur;
  }
  CUcontext ctx = CtxFor(tls_current_device);
  if (ctx != nullptr) nv::cuCtxSetCurrent(ctx);
  return ctx;
}

static cudaError_t ErrDbg(const char *fn, CUresult r) {
  if (r != CUDA_SUCCESS && getenv("DECORD_CUDART_DEBUG")) {
    fprintf(stderr, "[cudart-shim] %s -> driver error %d", fn, (int)r);
  }
  return r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInitializationError;
}
#define Err(r) ErrDbg(__FUNCTION__, r)

CUarray_format ChannelFormat(const cudaChannelFormatDesc &d) {
  if (d.f == cudaChannelFormatKindFloat) return CU_AD_FORMAT_FLOAT;
  if (d.f == cudaChannelFormatKindSigned) {
    return d.x > 8 ? CU_AD_FORMAT_SIGNED_INT16 : CU_AD_FORMAT_SIGNED_INT8;
  }
  return d.x > 8 ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT8;
}

int ChannelCount(const cudaChannelFormatDesc &d) {
  int n = 0;
  if (d.x) ++n;
  if (d.y) ++n;
  if (d.z) ++n;
  if (d.w) ++n;
  return n ? n : 1;
}

}  // namespace

cudaError_t cudaSetDevice(int device) {
  CUcontext ctx = CtxFor(device);
  if (!ctx) return cudaErrorInitializationError;
  cudaError_t r = Err(nv::cuCtxSetCurrent(ctx));
  if (r == cudaSuccess) tls_current_device = device;
  return r;
}

cudaError_t cudaGetDevice(int *device) {
  if (!device) return cudaErrorInvalidValue;
  *device = tls_current_device;
  return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr,
                                   int device) {
  /* runtime 枚举值 → 驱动 CU_DEVICE_ATTRIBUTE_*（多数一致，
   * MultiProcessorCount 例外：runtime 16 → 驱动 28） */
  CUdevice_attribute cu_attr;
  switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock:
      cu_attr = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK; break;
    case cudaDevAttrMaxBlockDimX:
      cu_attr = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X; break;
    case cudaDevAttrMaxBlockDimY:
      cu_attr = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y; break;
    case cudaDevAttrMaxBlockDimZ:
      cu_attr = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z; break;
    case cudaDevAttrMaxSharedMemoryPerBlock:
      cu_attr = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK; break;
    case cudaDevAttrWarpSize:
      cu_attr = CU_DEVICE_ATTRIBUTE_WARP_SIZE; break;
    case cudaDevAttrClockRate:
      cu_attr = CU_DEVICE_ATTRIBUTE_CLOCK_RATE; break;
    case cudaDevAttrMultiProcessorCount:
      cu_attr = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT; break;
    case cudaDevAttrComputeCapabilityMajor:
      cu_attr = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR; break;
    case cudaDevAttrComputeCapabilityMinor:
      cu_attr = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR; break;
    default:
      return cudaErrorInvalidValue;
  }
  CUcontext ctx = CtxFor(device);
  if (!ctx) return cudaErrorInitializationError;
  CUcontext prev = nullptr;
  nv::cuCtxPushCurrent(ctx);
  CUresult r = nv::cuDeviceGetAttribute(value, cu_attr, device);
  nv::cuCtxPopCurrent_v2(&prev);
  return Err(r);
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  EnsureCurrentCtx();
  return Err(nv::cuMemAlloc_v2(reinterpret_cast<CUdeviceptr *>(devPtr), size));
}

cudaError_t cudaFree(void *devPtr) {
  return Err(nv::cuMemFree_v2(reinterpret_cast<CUdeviceptr>(devPtr)));
}

cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
  EnsureCurrentCtx();
  return Err(nv::cuMemGetInfo_v2(free_bytes, total_bytes));
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
  EnsureCurrentCtx();
  (void)flags;  // cudaHostAllocDefault：非 portable pinned（单 primary ctx 等效）
  return Err(nv::cuMemHostAlloc(pHost, size, 0));
}

cudaError_t cudaMallocHost(void **pHost, size_t size) {
  return cudaHostAlloc(pHost, size, cudaHostAllocDefault);
}

cudaError_t cudaFreeHost(void *ptr) {
  return Err(nv::cuMemFreeHost(ptr));
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind) {
  EnsureCurrentCtx();
  switch (kind) {
    case cudaMemcpyHostToDevice:
      return Err(nv::cuMemcpyHtoD_v2(reinterpret_cast<CUdeviceptr>(dst), src,
                                  count));
    case cudaMemcpyDeviceToHost:
      return Err(nv::cuMemcpyDtoH_v2(dst, reinterpret_cast<CUdeviceptr>(src),
                                  count));
    case cudaMemcpyDeviceToDevice:
      return Err(nv::cuMemcpyDtoD_v2(reinterpret_cast<CUdeviceptr>(dst),
                                  reinterpret_cast<CUdeviceptr>(src), count));
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
  EnsureCurrentCtx();
  switch (kind) {
    case cudaMemcpyHostToDevice:
      return Err(nv::cuMemcpyHtoDAsync_v2(reinterpret_cast<CUdeviceptr>(dst),
                                       src, count,
                                       reinterpret_cast<CUstream>(stream)));
    case cudaMemcpyDeviceToHost:
      return Err(nv::cuMemcpyDtoHAsync_v2(dst, reinterpret_cast<CUdeviceptr>(src),
                                       count,
                                       reinterpret_cast<CUstream>(stream)));
    case cudaMemcpyDeviceToDevice:
      return Err(nv::cuMemcpyDtoDAsync_v2(reinterpret_cast<CUdeviceptr>(dst),
                                       reinterpret_cast<CUdeviceptr>(src),
                                       count,
                                       reinterpret_cast<CUstream>(stream)));
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags) {
  return Err(nv::cuStreamWaitEvent(reinterpret_cast<CUstream>(stream),
                                   reinterpret_cast<CUevent>(event), flags));
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                         size_t spitch, size_t width, size_t height,
                         enum cudaMemcpyKind kind) {
  EnsureCurrentCtx();
  /* 本项目调用均为 host↔device 一维行拷贝（kind 区分方向） */
  CUDA_MEMCPY2D m = {};
  if (kind == cudaMemcpyHostToDevice) {
    m.srcMemoryType = CU_MEMORYTYPE_HOST;
    m.srcHost = src;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
  } else {
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = reinterpret_cast<CUdeviceptr>(const_cast<void *>(src));
    m.dstMemoryType = CU_MEMORYTYPE_HOST;
    m.dstHost = dst;
  }
  m.srcPitch = spitch;
  m.dstPitch = dpitch;
  m.WidthInBytes = width;
  m.Height = height;
  return Err(nv::cuMemcpy2D_v2(&m));
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count,
                                cudaStream_t stream) {
  CUcontext dctx = CtxFor(dstDevice);
  CUcontext sctx = CtxFor(srcDevice);
  if (!dctx || !sctx) return cudaErrorInitializationError;
  return Err(nv::cuMemcpyPeerAsync_v2(
      reinterpret_cast<CUdeviceptr>(dst), dctx,
      reinterpret_cast<CUdeviceptr>(const_cast<void *>(src)), sctx, count,
      reinterpret_cast<CUstream>(stream)));
}

cudaError_t cudaStreamCreate(cudaStream_t *stream) {
  EnsureCurrentCtx();
  return Err(nv::cuStreamCreate(reinterpret_cast<CUstream *>(stream),
                                CU_STREAM_DEFAULT));
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,
                                      unsigned int flags) {
  EnsureCurrentCtx();
  return Err(nv::cuStreamCreate(reinterpret_cast<CUstream *>(stream), flags));
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return Err(nv::cuStreamSynchronize(reinterpret_cast<CUstream>(stream)));
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return Err(nv::cuStreamDestroy(reinterpret_cast<CUstream>(stream)));
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
  EnsureCurrentCtx();
  return Err(nv::cuEventCreate(reinterpret_cast<CUevent *>(event), 0));
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  EnsureCurrentCtx();
  return Err(nv::cuEventCreate(reinterpret_cast<CUevent *>(event), flags));
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return Err(nv::cuEventRecord(reinterpret_cast<CUevent>(event),
                               reinterpret_cast<CUstream>(stream)));
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return Err(nv::cuEventSynchronize(reinterpret_cast<CUevent>(event)));
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return Err(nv::cuEventDestroy(reinterpret_cast<CUevent>(event)));
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                                    const struct cudaResourceDesc *pResDesc,
                                    const struct cudaTextureDesc *pTexDesc,
                                    const struct cudaResourceViewDesc *) {
  EnsureCurrentCtx();
  CUDA_RESOURCE_DESC rd = {};
  rd.resType = CU_RESOURCE_TYPE_PITCH2D;
  if (pResDesc->resType != cudaResourceTypePitch2D) {
    return cudaErrorInvalidValue;  // 本项目只用 Pitch2D（cuvid 映射帧）
  }
  rd.res.pitch2D.devPtr = reinterpret_cast<CUdeviceptr>(
      pResDesc->res.pitch2D.devPtr);
  rd.res.pitch2D.format = ChannelFormat(pResDesc->res.pitch2D.desc);
  rd.res.pitch2D.numChannels = ChannelCount(pResDesc->res.pitch2D.desc);
  rd.res.pitch2D.width = pResDesc->res.pitch2D.width;
  rd.res.pitch2D.height = pResDesc->res.pitch2D.height;
  rd.res.pitch2D.pitchInBytes = pResDesc->res.pitch2D.pitchInBytes;

  CUDA_TEXTURE_DESC td = {};
  for (int i = 0; i < 3; ++i) {
    /* runtime 与驱动地址模式数值一致（WRAP0/CLAMP1/MIRROR2/BORDER3） */
    td.addressMode[i] = static_cast<CUaddress_mode>(pTexDesc->addressMode[i]);
  }
  td.filterMode = static_cast<CUfilter_mode>(pTexDesc->filterMode);
  if (pTexDesc->readMode == cudaReadModeElementType) {
    td.flags |= CU_TRSF_READ_AS_INTEGER;
  }
  if (pTexDesc->normalizedCoords) {
    td.flags |= CU_TRSF_NORMALIZED_COORDINATES;
  }
  td.maxAnisotropy = pTexDesc->maxAnisotropy;
  return Err(nv::cuTexObjectCreate(reinterpret_cast<CUtexObject *>(pTexObject),
                                   &rd, &td, nullptr));
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
  return Err(nv::cuTexObjectDestroy(texObject));
}

const char *cudaGetErrorString(cudaError_t error) {
  if (error == cudaSuccess) return "no error";
  const char *msg = nullptr;
  if (nv::cuGetErrorString(static_cast<CUresult>(error), &msg) == CUDA_SUCCESS
          && msg != nullptr) {
    return msg;
  }
  return "unknown cudaError";
}

/* 全局转发：历史代码（cuda_common.h 的 check_cuda_call）以非限定名调用
 * 这两个函数——旧 toolchain 下由 cuda.h 声明 + cudart 定义；现在由 dyn
 * 包装器提供定义。 */
CUresult cuGetErrorString(CUresult error, const char **pStr) {
  return nv::cuGetErrorString(error, pStr);
}
CUresult cuGetErrorName(CUresult error, const char **pStr) {
  return nv::cuGetErrorName(error, pStr);
}
