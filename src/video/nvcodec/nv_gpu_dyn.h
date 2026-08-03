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

namespace nv {

/*! \brief Whether GPU symbols were loaded (NVIDIA driver present). Loads on first call. */
bool gpu_loaded();

/* -- CUDA driver API (nvcuda.dll; old drivers forward via nvcuvid.dll) -- */
CUresult CUDAAPI cuInit(unsigned int flags);
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

/* -- NVML (nvml.dll, driver version probe only; absence does not block GPU) -- */
nvmlReturn_t CUDAAPI nvmlInit(void);
nvmlReturn_t CUDAAPI nvmlSystemGetDriverVersion(char* version, unsigned int length);

}  // namespace nv
