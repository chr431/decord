/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file improc.h
 * \brief Image processing functions
 */

#ifndef DECORD_IMPROC_IMPROC_H_
#define DECORD_IMPROC_IMPROC_H_

#include <stdint.h>

#include "../runtime/cuda/cudart_shim.h"

namespace decord {
namespace cuda {

#ifdef DECORD_USE_CUDA

// 返回 false = kernel 启动失败(模块加载/launch 错误)。不抛异常:本函数在
// CUVID display 回调(驱动栈帧)内运行,异常不得穿越驱动帧展开;调用方
// (cuda_threaded_decoder)收到 false 后 RecordInternalError + 丢弃该帧,
// 由下一次 Pop() 统一转成 Python 可见异常。
bool ProcessFrame(cudaTextureObject_t chroma, cudaTextureObject_t luma,
                  uint8_t* dst, cudaStream_t stream, uint16_t input_width, uint16_t input_height,
                  int output_width, int output_height,
                  int src_x0 = 0, int src_y0 = 0,
                  float fx = 0.0f, float fy = 0.0f,
                  int bit_depth = 8,
                  int output_format = 0,
                  int color_range = 0);
#endif
}  // namespace imp
}  // namespace decord


#endif  // DECORD_IMPROC_IMPROC_H_
