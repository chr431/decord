/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file improc_dyn.cc
 * \brief GPU 图像处理（NV12 → RGB/gray/YUV420 + ROI + 缩放）——驱动 API
 *  版本：improc kernel 由预编译 PTX（improc_ptx.inc，nvcc -ptx
 *  -arch=compute_75 一次性生成）经 cuModuleLoadData 加载，
 *  cuLaunchKernel 启动。构建不再需要 CUDA Toolkit / nvcc。
 *
 *  kernel 源码与再生成：
 *    src/improc/improc.cu（仅再生成 PTX 时需要 nvcc）
 *    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
 *      -ptx -arch=compute_75 -I src/improc src/improc/improc.cu
 *      -o src/improc/improc.ptx
 *    然后 python 重生成 improc_ptx.inc（见仓库脚本/历史 commit）。
 *
 *  PTX 兼容性：compute_75 的 PTX 可被后续所有架构 JIT（驱动在上下文
 *  初始化后按需编译一次并缓存）。
 */
#include "improc.h"

#include <mutex>
#include <dmlc/logging.h>

#include "../video/nvcodec/nv_gpu_dyn.h"
#include "../runtime/cuda/cudart_shim.h"
#include "../runtime/cuda/cuda_common.h"

#include "improc_ptx.inc"

namespace decord {
namespace cuda {
namespace {

const char kKernelName[] =
    "_ZN6decord4cuda6detail20process_frame_kernelIhEEvyyPT_ttttiiffii";

std::once_flag g_once;
bool g_ready = false;
CUmodule g_module = nullptr;
CUfunction g_function = nullptr;

void InitModule() {
  if (!nv::gpu_loaded()) {
    LOG(FATAL) << "improc: NVIDIA driver not available";
  }
  if (nv::cuModuleLoadData(&g_module, kImprocPtx) != CUDA_SUCCESS) {
    LOG(FATAL) << "improc: failed to load embedded PTX module";
  }
  if (nv::cuModuleGetFunction(&g_function, g_module, kKernelName)
          != CUDA_SUCCESS) {
    LOG(FATAL) << "improc: kernel entry not found in PTX: " << kKernelName;
  }
  g_ready = true;
}

int DivUp(int total, int grain) {
  return (total + grain - 1) / grain;
}

}  // namespace

void ProcessFrame(cudaTextureObject_t chroma, cudaTextureObject_t luma,
    uint8_t* dst, cudaStream_t stream, uint16_t input_width, uint16_t input_height,
    int output_width, int output_height,
    int src_x0, int src_y0, float fx, float fy, int bit_depth,
    int output_format, int color_range) {
  // resize factor: 0 表示由本函数按 in/out 推导（全帧缩放路径）；
  // ROI-first 路径由调用方传 1.0（窗口像素 1:1 映射）。
  if (fx <= 0.0f) fx = static_cast<float>(input_width) / output_width;
  if (fy <= 0.0f) fy = static_cast<float>(input_height) / output_height;
  // 位深语义：P016/P012 的 10/12-bit 数据在 16-bit 字中左对齐存储，
  // normalized float 采样即等价（见 improc.cu 注释）。
  (void)bit_depth;

  std::call_once(g_once, InitModule);
  if (!g_ready) return;  // InitModule 已 LOG(FATAL)，防御

  // kernel 形参顺序：luma(tex) chroma(tex) dst in_w in_h out_w out_h
  //                 src_x0 src_y0 fx fy output_format color_range
  unsigned long long luma_h = luma, chroma_h = chroma;
  uint16_t inw = input_width, inh = input_height;
  uint16_t outw = static_cast<uint16_t>(output_width);
  uint16_t outh = static_cast<uint16_t>(output_height);
  void* params[] = {
      &luma_h, &chroma_h, &dst, &inw, &inh, &outw, &outh,
      &src_x0, &src_y0, &fx, &fy, &output_format, &color_range};

  CUresult lr = nv::cuLaunchKernel(
      g_function, DivUp(output_width, 32), DivUp(output_height, 8), 1,
      32, 8, 1, 0, reinterpret_cast<CUstream>(stream), params, nullptr);
  if (lr != CUDA_SUCCESS) {
    LOG(FATAL) << "improc: cuLaunchKernel failed: " << lr;
  }
}

}  // namespace cuda
}  // namespace decord
