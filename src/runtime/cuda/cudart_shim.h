/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file cudart_shim.h
 * \brief CUDA Runtime API 垫片：同名/同签名的类型与函数声明，实现走
 *  nv:: 动态加载的 CUDA 驱动 API（nv_gpu_dyn）。
 *
 * 0.8.1 起 CUDA 构建不再需要 CUDA Toolkit（无 nvcc / 无 cudart 库 /
 * 无 toolkit 头文件）——之前唯一需要 nvcc 的 improc kernel 已改为
 * 预编译 PTX + cuModuleLoadData 加载（src/improc/improc_dyn.cc）。
 *
 * 语义注意：
 * - cudaSetDevice = 保留该设备的 primary context 并置为当前（per-thread，
 *   与 runtime API 一致；与 cuda_context.cc 的 primary-ctx 模型兼容）。
 * - 同步 cudaMemcpy：驱动在 pageable 内存上内部 staging（与 runtime
 *   行为一致）；cudaMemcpyAsync(0 流) = legacy 默认流（同步语义一致）。
 * - cudaHostAllocDefault = 非 portable pinned：单 primary-ctx 使用下与
 *   runtime 等效。
 * - 事件/流创建标志位与 runtime 数值一致（DISABLE_TIMING=0x02、
 *   NonBlocking=0x01），直接透传。
 */
#ifndef DECORD_RUNTIME_CUDA_CUDART_SHIM_H_
#define DECORD_RUNTIME_CUDA_CUDART_SHIM_H_

/* 与真实 runtime 头一致的可用性标记：cuda_common.h 的 check_cuda_call
 * runtime 重载靠它启用 */
#ifndef __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_H__
#endif

#include <cuda.h>   // vendored: src/video/nvcodec/cuda_include/cuda.h
#include <cstddef>
#include <stdint.h>

/* ── 错误码 ── */
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorCudartUnloading 5
#define cudaErrorInitializationError 3
#define cudaErrorInvalidValue 12
#define cudaErrorInsufficientDriver 35

/* ── 句柄：与驱动句柄二进制兼容（cudart 本就是驱动 API 的薄封装） ── */
typedef CUstream cudaStream_t;
typedef CUevent cudaEvent_t;
typedef unsigned long long cudaTextureObject_t;

/* ── memcpy kind ── */
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

/* ── 流/事件创建标志 ── */
enum { cudaStreamDefault = 0x00, cudaStreamNonBlocking = 0x01 };
enum {
  cudaEventDefault = 0x00,
  cudaEventBlockingSync = 0x01,
  cudaEventDisableTiming = 0x02
};
enum { cudaHostAllocDefault = 0x00 };

/* ── 设备属性（runtime 枚举值；实现里映射到 CU_DEVICE_ATTRIBUTE_*） ── */
enum cudaDeviceAttr {
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxBlockDimX = 2,
  cudaDevAttrMaxBlockDimY = 3,
  cudaDevAttrMaxBlockDimZ = 4,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrWarpSize = 10,
  cudaDevAttrClockRate = 13,
  cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76
};

/* ── 纹理通道/资源/采样描述（runtime 布局；实现里转换为驱动结构） ── */
enum cudaChannelFormatKind {
  cudaChannelFormatKindSigned = 0,
  cudaChannelFormatKindUnsigned = 1,
  cudaChannelFormatKindFloat = 2,
  cudaChannelFormatKindNone = 7
};
struct cudaChannelFormatDesc {
  int x, y, z, w;
  enum cudaChannelFormatKind f;
};
struct cudaArray;
enum cudaResourceType {
  cudaResourceTypeArray = 0,
  cudaResourceTypeMipmappedArray = 1,
  cudaResourceTypeLinear = 2,
  cudaResourceTypePitch2D = 3
};
struct cudaResourceDesc {
  enum cudaResourceType resType;
  union {
    struct { cudaArray *array; } array;
    struct {
      void *devPtr;
      struct cudaChannelFormatDesc desc;
      size_t sizeInBytes;
    } linear;
    struct {
      void *devPtr;
      struct cudaChannelFormatDesc desc;
      size_t width;
      size_t height;
      size_t pitchInBytes;
    } pitch2D;
  } res;
};
enum cudaTextureAddressMode {
  cudaAddressModeWrap = 0,
  cudaAddressModeClamp = 1,
  cudaAddressModeMirror = 2,
  cudaAddressModeBorder = 3
};
enum cudaTextureFilterMode {
  cudaFilterModePoint = 0,
  cudaFilterModeLinear = 1
};
enum cudaTextureReadMode {
  cudaReadModeElementType = 0,
  cudaReadModeNormalizedFloat = 1
};
struct cudaTextureDesc {
  enum cudaTextureAddressMode addressMode[3];
  enum cudaTextureFilterMode filterMode;
  enum cudaTextureReadMode readMode;
  int normalizedCoords;
  unsigned int maxAnisotropy;
};
struct cudaResourceViewDesc {};  /* runtime 调用方恒传 nullptr */

/* ── 向量类型（cudaCreateChannelDesc<T> 模板实参） ── */
struct uchar1 { uint8_t x; };
struct uchar2 { uint8_t x, y; };
struct ushort1 { uint16_t x; };
struct ushort2 { uint16_t x, y; };

/* ── 函数（drop-in，与 runtime 同名同签名） ── */
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr,
                                   int device);
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes);
cudaError_t cudaMallocHost(void **pHost, size_t size);
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                         size_t spitch, size_t width, size_t height,
                         enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count,
                                cudaStream_t stream);
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,
                                      unsigned int flags);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                                    const struct cudaResourceDesc *pResDesc,
                                    const struct cudaTextureDesc *pTexDesc,
                                    const struct cudaResourceViewDesc *pResViewDesc);
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
const char *cudaGetErrorString(cudaError_t error);

/* channel desc（与 runtime 内置函数等价的便捷模板） */
template <typename T>
inline cudaChannelFormatDesc cudaCreateChannelDesc();

template <>
inline cudaChannelFormatDesc cudaCreateChannelDesc<uchar1>() {
  cudaChannelFormatDesc d = {8, 0, 0, 0, cudaChannelFormatKindUnsigned};
  return d;
}
template <>
inline cudaChannelFormatDesc cudaCreateChannelDesc<uchar2>() {
  cudaChannelFormatDesc d = {8, 8, 0, 0, cudaChannelFormatKindUnsigned};
  return d;
}
template <>
inline cudaChannelFormatDesc cudaCreateChannelDesc<ushort1>() {
  cudaChannelFormatDesc d = {16, 0, 0, 0, cudaChannelFormatKindUnsigned};
  return d;
}
template <>
inline cudaChannelFormatDesc cudaCreateChannelDesc<ushort2>() {
  cudaChannelFormatDesc d = {16, 16, 0, 0, cudaChannelFormatKindUnsigned};
  return d;
}

#endif  // DECORD_RUNTIME_CUDA_CUDART_SHIM_H_
