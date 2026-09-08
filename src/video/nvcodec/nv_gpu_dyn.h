/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file nv_gpu_dyn.h
 * \brief Dynamically load NVIDIA GPU APIs (CUDA driver API / NVCUVID / NVML)
 *
 * GPU decord statically links nvml.lib / nvcuvid.lib, so decord.dll imports
 * nvml.dll / nvcuvid.dll directly (both provided by the NVIDIA driver); on
 * driver-less machines the whole DLL fails to load, killing CPU decode too.
 *
 * This module switches to runtime LoadLibrary / GetProcAddress:
 *   - - with NVIDIA driver: functions resolve, GPU decode behaves identically;
 *   - - without driver: LoadLibrary fails, wrappers return error codes
 *     (CUDA_ERROR_NOT_FOUND / NVML_ERROR_LIBRARY_NOT_FOUND) so the upper
 *     CHECK_CUDA_CALL raises -> Python falls back to CPU decode.
 */
#pragma once

#include <cuda.h>
#include <nvml.h>
#include <nvcuvid.h>

/* cuda.h(13.x) 把内存/拷贝类 API 宏重定向到 *_v2 旧符号，而新驱动已
 * 移除 v2 导出 —— 统一 #undef，按非后缀名加载（CUDA 12+ 驱动导出）。 */
#ifdef cuMemAlloc
#undef cuMemAlloc
#endif
#ifdef cuMemFree
#undef cuMemFree
#endif
#ifdef cuMemGetInfo
#undef cuMemGetInfo
#endif
#ifdef cuMemcpyHtoD
#undef cuMemcpyHtoD
#endif
#ifdef cuMemcpyDtoH
#undef cuMemcpyDtoH
#endif
#ifdef cuMemcpyDtoD
#undef cuMemcpyDtoD
#endif
#ifdef cuMemcpyHtoDAsync
#undef cuMemcpyHtoDAsync
#endif
#ifdef cuMemcpyDtoHAsync
#undef cuMemcpyDtoHAsync
#endif
#ifdef cuMemcpyDtoDAsync
#undef cuMemcpyDtoDAsync
#endif
#ifdef cuMemcpyPeerAsync
#undef cuMemcpyPeerAsync
#endif
#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif
#ifdef cuEventDestroy
#undef cuEventDestroy
#endif
#ifdef cuMemcpy2D
#undef cuMemcpy2D
#endif
/* cuda.h 会把 cuMemcpy2D 宏展开为 _v2；本仓库统一用非后缀符号 */
#ifdef cuMemcpy2D
#undef cuMemcpy2D
#endif

namespace nv {

/*! \brief Whether GPU symbols were loaded (NVIDIA driver present). Loads on first call. */
bool gpu_loaded();

/* -- CUDA driver API (nvcuda.dll; old drivers forward via nvcuvid.dll) -- */
CUresult CUDAAPI cuInit(unsigned int flags);
CUresult CUDAAPI cuGetErrorString(CUresult error, const char** pStr);
CUresult CUDAAPI cuGetErrorName(CUresult error, const char** pStr);
CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal);
CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx);
CUresult CUDAAPI cuCtxGetCurrent(CUcontext* pctx);
CUresult CUDAAPI cuCtxGetDevice(CUdevice* device);
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext* pctx);  /* cuCtxPopCurrent macro expands to _v2 */
CUresult CUDAAPI cuCtxSynchronize(void);
CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);

/* -- CUDA driver API: memory / copy / stream / event / device attr / tex /
      module+launch（cudart_shim 与 improc PTX 加载用；0.8.1 起 CUDA 构建
      不再链接 cudart、不再需要 Toolkit） -- */
CUresult CUDAAPI cuMemGetInfo_v2(size_t* free, size_t* total);
CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr);
CUresult CUDAAPI cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags);
CUresult CUDAAPI cuMemFreeHost(void* p);
CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src,
                                   size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr src,
                                   size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src,
                                   size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyPeerAsync_v2(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D* pCopy);
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags);
CUresult CUDAAPI cuStreamCreate(CUstream* pStream, unsigned int Flags);
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream);
CUresult CUDAAPI cuStreamDestroy(CUstream hStream);
CUresult CUDAAPI cuEventCreate(CUevent* phEvent, unsigned int Flags);
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult CUDAAPI cuEventSynchronize(CUevent hEvent);
CUresult CUDAAPI cuEventDestroy(CUevent hEvent);
CUresult CUDAAPI cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib,
                                      CUdevice dev);
CUresult CUDAAPI cuTexObjectCreate(CUtexObject* pTexObject,
                                   const CUDA_RESOURCE_DESC* pResDesc,
                                   const CUDA_TEXTURE_DESC* pTexDesc,
                                   const CUDA_RESOURCE_VIEW_DESC* pResViewDesc);
CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject);
CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image);
CUresult CUDAAPI cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod,
                                     const char* name);
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void** kernelParams, void** extra);

/* -- NVCUVID (nvcuvid.dll) -- */
CUresult CUDAAPI cuvidCreateDecoder(CUvideodecoder* phDecoder, CUVIDDECODECREATEINFO* pdci);
CUresult CUDAAPI cuvidDestroyDecoder(CUvideodecoder hDecoder);
CUresult CUDAAPI cuvidCreateVideoParser(CUvideoparser* phParser, CUVIDPARSERPARAMS* pParams);
CUresult CUDAAPI cuvidDestroyVideoParser(CUvideoparser hParser);
CUresult CUDAAPI cuvidParseVideoData(CUvideoparser hParser, CUVIDSOURCEDATAPACKET* pPacket);
CUresult CUDAAPI cuvidDecodePicture(CUvideodecoder hDecoder, CUVIDPICPARAMS* pPicParams);
CUresult CUDAAPI cuvidGetDecoderCaps(CUVIDDECODECAPS* pdc);
CUresult CUDAAPI cuvidMapVideoFrame(CUvideodecoder hDecoder, int nPicIdx,
                            unsigned int* pDevPtr, unsigned int* pPitch,
                            CUVIDPROCPARAMS* pVPP);
CUresult CUDAAPI cuvidUnmapVideoFrame(CUvideodecoder hDecoder, unsigned int DevPtr);
/* 64-bit pointer variants (used via __CUVID_DEVPTR64 macro expansion) */
CUresult CUDAAPI cuvidMapVideoFrame64(CUvideodecoder hDecoder, int nPicIdx,
                              unsigned long long* pDevPtr, unsigned int* pPitch,
                              CUVIDPROCPARAMS* pVPP);
CUresult CUDAAPI cuvidUnmapVideoFrame64(CUvideodecoder hDecoder, unsigned long long DevPtr);

/* -- NVML (nvml.dll, driver version probe only; absence does not block GPU) --
   nvml.h declares these DECLDIR (__cdecl), not __stdcall/CUDAAPI.  On x64 the
   calling conventions are identical, but match the real prototypes anyway. */
nvmlReturn_t nvmlInit(void);
nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length);

}  // namespace nv
