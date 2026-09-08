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

// 模块缓存按"加载时所在上下文"键控:module/function 句柄是上下文作用域的,
// 主上下文若被销毁重建(refcount 归零 / 外部 reset),旧句柄全部悬空,再用
// 即 cuLaunchKernel 失败乃至驱动内访问违例。检测到上下文变化就重新加载
// (驱动有磁盘 JIT 缓存,重载开销为一次性数十毫秒)。
std::mutex g_mu;
CUcontext g_ctx = nullptr;
CUmodule g_module = nullptr;
CUfunction g_function = nullptr;
bool g_ready = false;

bool InitModuleLocked(CUcontext cur) {
  if (!nv::gpu_loaded()) {
    fprintf(stderr, "[improc] NVIDIA driver not available\n");
    return false;
  }
  if (nv::cuModuleLoadData(&g_module, kImprocPtx) != CUDA_SUCCESS) {
    fprintf(stderr, "[improc] failed to load embedded PTX module\n");
    return false;
  }
  if (nv::cuModuleGetFunction(&g_function, g_module, kKernelName)
          != CUDA_SUCCESS) {
    fprintf(stderr, "[improc] kernel entry not found in PTX: %s\n", kKernelName);
    return false;
  }
  g_ctx = cur;
  g_ready = true;
  return true;
}

int DivUp(int total, int grain) {
  return (total + grain - 1) / grain;
}

}  // namespace

bool ProcessFrame(cudaTextureObject_t chroma, cudaTextureObject_t luma,
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

  CUcontext cur = nullptr;
  nv::cuCtxGetCurrent(&cur);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_ready || cur != g_ctx) {
      if (!InitModuleLocked(cur)) return false;
    }
  }

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
    // 不抛异常(本函数运行于 CUVID display 回调的驱动栈帧内,异常穿越驱动帧
    // 的行为不受我们控制);报告失败由调用方丢弃该帧并记录内部错误。
    fprintf(stderr, "[improc] cuLaunchKernel failed: %d\n", static_cast<int>(lr));
    {
      // 启动失败通常意味着句柄/上下文异常:作废缓存,下次强制重载
      std::lock_guard<std::mutex> lk(g_mu);
      g_ready = false;
    }
    return false;
  }
  return true;
}

}  // namespace cuda
}  // namespace decord
