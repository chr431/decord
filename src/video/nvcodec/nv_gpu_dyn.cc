/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file nv_gpu_dyn.cc
 * \brief Implementation of dynamically loaded NVIDIA GPU APIs (see nv_gpu_dyn.h)
 */

#include "nv_gpu_dyn.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nv {

namespace {

/* Shared-library handle + load/symbol helpers (Windows vs dlopen) */
#if defined(_WIN32)
using nv_lib_t = HMODULE;
#define NV_LOAD_LIB(name) LoadLibraryA(name)
#define NV_GET_SYM(handle, name) GetProcAddress((handle), (name))
#else
using nv_lib_t = void*;
#define NV_LOAD_LIB(name) dlopen((name), RTLD_LAZY)
#define NV_GET_SYM(handle, name) dlsym((handle), (name))
#endif

/* Function list macro: X(ret, default_err, name, params, call_args) */
#define NV_CU_FUNCS(X)                                                          \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuInit, (unsigned int flags), (flags))      \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuDeviceGet,                                \
    (CUdevice* device, int ordinal), (device, ordinal))                         \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuDeviceGetName,                            \
    (char* name, int len, CUdevice dev), (name, len, dev))                      \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuCtxDestroy, (CUcontext ctx), (ctx))       \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuCtxGetCurrent, (CUcontext* pctx), (pctx)) \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuCtxGetDevice, (CUdevice* device),         \
    (device))                                                                   \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuCtxPushCurrent, (CUcontext ctx), (ctx))   \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuCtxSetCurrent, (CUcontext ctx), (ctx))     \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuCtxPopCurrent_v2, (CUcontext* pctx),      \
    (pctx))                                                                      \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuCtxSynchronize, (void), ())               \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuDevicePrimaryCtxRelease,                  \
    (CUdevice dev), (dev))                                                      \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuDevicePrimaryCtxRetain,                   \
    (CUcontext* pctx, CUdevice dev), (pctx, dev))                               \
  /* NVCUVID */                                                                 \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidCreateDecoder,                         \
    (CUvideodecoder* phDecoder, CUVIDDECODECREATEINFO* pdci),                   \
    (phDecoder, pdci))                                                          \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidDestroyDecoder,                        \
    (CUvideodecoder hDecoder), (hDecoder))                                      \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidCreateVideoParser,                     \
    (CUvideoparser* phParser, CUVIDPARSERPARAMS* pParams), (phParser, pParams)) \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidDestroyVideoParser,                    \
    (CUvideoparser hParser), (hParser))                                         \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidParseVideoData,                        \
    (CUvideoparser hParser, CUVIDSOURCEDATAPACKET* pPacket), (hParser, pPacket))\
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidDecodePicture,                         \
    (CUvideodecoder hDecoder, CUVIDPICPARAMS* pPicParams),                      \
    (hDecoder, pPicParams))                                                     \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidGetDecoderCaps,                        \
    (CUVIDDECODECAPS* pdc), (pdc))                                              \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidMapVideoFrame,                         \
    (CUvideodecoder hDecoder, int nPicIdx, unsigned int* pDevPtr,               \
     unsigned int* pPitch, CUVIDPROCPARAMS* pVPP),                              \
    (hDecoder, nPicIdx, pDevPtr, pPitch, pVPP))                                 \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidUnmapVideoFrame,                       \
    (CUvideodecoder hDecoder, unsigned int DevPtr), (hDecoder, DevPtr))         \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidMapVideoFrame64,                       \
    (CUvideodecoder hDecoder, int nPicIdx, unsigned long long* pDevPtr,         \
     unsigned int* pPitch, CUVIDPROCPARAMS* pVPP),                              \
    (hDecoder, nPicIdx, pDevPtr, pPitch, pVPP))                                 \
  X(CUresult, CUDA_ERROR_NOT_FOUND, cuvidUnmapVideoFrame64,                     \
    (CUvideodecoder hDecoder, unsigned long long DevPtr), (hDecoder, DevPtr))

#define NV_NVML_FUNCS(X)                                                        \
  X(nvmlReturn_t, NVML_ERROR_LIBRARY_NOT_FOUND, nvmlInit, (void), ())           \
  X(nvmlReturn_t, NVML_ERROR_LIBRARY_NOT_FOUND, nvmlSystemGetDriverVersion,     \
    (char* version, unsigned int length), (version, length))

/* function pointers */
#define NV_DECL_PTR(ret, err, name, args, callargs) \
  static ret (CUDAAPI *p_##name) args = nullptr;
NV_CU_FUNCS(NV_DECL_PTR)
NV_NVML_FUNCS(NV_DECL_PTR)
#undef NV_DECL_PTR

std::once_flag g_load_once;
bool g_loaded = false;

}  // namespace

bool gpu_loaded() {
  std::call_once(g_load_once, []() {
  /* Test hook: DECORD_FORCE_NO_GPU=1 simulates a machine without the NVIDIA
     driver (skips loading nvcuda/nvcuvid/nvml). Same effect as the real
     driver-less case: wrappers return error codes -> GPU init fails ->
     Python falls back to CPU decode. */
  const char* force_no_gpu = std::getenv("DECORD_FORCE_NO_GPU");
  if (force_no_gpu && std::strcmp(force_no_gpu, "1") == 0) {
    g_loaded = false;
    return;
  }

#if defined(_WIN32)
  nv_lib_t nvcuda = NV_LOAD_LIB("nvcuda.dll");
  nv_lib_t nvcuvid = NV_LOAD_LIB("nvcuvid.dll");
  nv_lib_t nvml = NV_LOAD_LIB("nvml.dll");
#else
  nv_lib_t nvcuda = NV_LOAD_LIB("libcuda.so.1");
  nv_lib_t nvcuvid = NV_LOAD_LIB("libnvcuvid.so");
  nv_lib_t nvml = NV_LOAD_LIB("libnvidia-ml.so.1");
#endif

  /* cu* and cuvid*: try nvcuda.dll first, then nvcuvid.dll
     (old drivers lack nvcuda.dll; driver API forwarded via nvcuvid.dll) */
#define NV_LOAD_SYM(ret, err, name, args, callargs)          \
  p_##name = reinterpret_cast<decltype(p_##name)>(           \
      NV_GET_SYM(nvcuda, #name));                            \
  if (!p_##name && nvcuvid)                                  \
    p_##name = reinterpret_cast<decltype(p_##name)>(         \
        NV_GET_SYM(nvcuvid, #name));
  NV_CU_FUNCS(NV_LOAD_SYM)
#undef NV_LOAD_SYM

#define NV_LOAD_SYM_NVML(ret, err, name, args, callargs)     \
  p_##name = reinterpret_cast<decltype(p_##name)>(           \
      NV_GET_SYM(nvml, #name));
  NV_NVML_FUNCS(NV_LOAD_SYM_NVML)
#undef NV_LOAD_SYM_NVML

  /* usable when cuInit + all cuvid are ready (NVML is probe-only) */
  g_loaded = p_cuInit != nullptr && p_cuvidCreateDecoder != nullptr;
  });
  return g_loaded;
}

/* wrappers: not loaded -> error code (CHECK_CUDA_CALL raises -> Python fallback) */
#define NV_WRAP(ret, err, name, args, callargs)              \
  ret CUDAAPI name args {                                            \
    gpu_loaded();                                            \
    if (!p_##name) return err;                               \
    return p_##name callargs;                                \
  }
NV_CU_FUNCS(NV_WRAP)
NV_NVML_FUNCS(NV_WRAP)
#undef NV_WRAP

}  // namespace nv
