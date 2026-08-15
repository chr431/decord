/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file improc.cu
 * \brief CUDA image processing kernels
 */

#include "improc.h"
#include <cuda_fp16.h>
// #include <stdio.h>

namespace decord {
namespace cuda {
namespace detail {

template<typename T>
struct YUV {
    T y, u, v;
};

__constant__ float yuv2rgb_mat[9] = {
    1.164383f,  0.0f,       1.596027f,
    1.164383f, -0.391762f, -0.812968f,
    1.164383f,  2.017232f,  0.0f
};

__device__ float clip(float x, float max) {
    return fmin(fmax(x, 0.0f), max);
}

template<typename T>
__device__ T convert(const float x) {
    return static_cast<T>(x);
}

template<>
__device__ half convert<half>(const float x) {
    return __float2half(x);
}

template<>
__device__ uint8_t convert<uint8_t>(const float x) {
    return static_cast<uint8_t>(roundf(x));
}

template<typename YUV_T, typename RGB_T>
__device__ void yuv2rgb(const YUV<YUV_T>& yuv, RGB_T* rgb,
                        size_t stride, bool normalized) {
    auto mult = normalized ? 1.0f : 255.0f;
    auto y = (static_cast<float>(yuv.y) - 16.0f/255) * mult;
    auto u = (static_cast<float>(yuv.u) - 128.0f/255) * mult;
    auto v = (static_cast<float>(yuv.v) - 128.0f/255) * mult;

    auto& m = yuv2rgb_mat;

    // could get tricky with a lambda, but this branch seems faster
    float r, g, b;
    if (normalized) {
        r = clip(y*m[0] + u*m[1] + v*m[2], 1.0);
        g = clip(y*m[3] + u*m[4] + v*m[5], 1.0);
        b = clip(y*m[6] + u*m[7] + v*m[8], 1.0);
    } else {
        r = clip(y*m[0] + u*m[1] + v*m[2], 255.0);
        g = clip(y*m[3] + u*m[4] + v*m[5], 255.0);
        b = clip(y*m[6] + u*m[7] + v*m[8], 255.0);
    }

    rgb[0] = convert<RGB_T>(r);
    rgb[stride] = convert<RGB_T>(g);
    rgb[stride*2] = convert<RGB_T>(b);
}

template<typename T>
__global__ void process_frame_kernel(
    cudaTextureObject_t luma, cudaTextureObject_t chroma,
    T* dst, uint16_t input_width, uint16_t input_height,
    uint16_t output_width, uint16_t output_height,
    int src_x0, int src_y0, float fx, float fy,
    int output_format, int color_range) {

    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= output_width || dst_y >= output_height)
        return;

    auto src_x = (static_cast<float>(dst_x) + static_cast<float>(src_x0)) * fx;

    auto src_y = (static_cast<float>(dst_y) + static_cast<float>(src_y0)) * fy;

    YUV<float> yuv;
    yuv.y = tex2D<float>(luma, src_x + 0.5, src_y + 0.5);
    if (output_format == 1) {
        // GRAY8（= Y 平面，按流 range 语义）——与 CPU 路径 swscale 的
        // GRAY8 输出一致（两侧都遵循流的 color_range）：
        //   limited (tv): 展开 (Y-16)*255/219（BT.601 limited->full）
        //   full (pc):    原始 Y 原样
        // 灰度路径不需要色度采样（省一次 chroma 纹理访问）。
        // color_range 语义：0 = limited/tv（展开），1 = full/pc（原样）。
        if (color_range == 0) {
            dst[dst_x + dst_y * output_width] =
                convert<T>(clip((yuv.y - 16.0f / 255.0f)
                                    * (255.0f / 219.0f) * 255.0f,
                                255.0f));
        } else {
            dst[dst_x + dst_y * output_width] =
                convert<T>(clip(yuv.y * 255.0f, 255.0f));
        }
        return;
    }
    // 4:2:0 chroma siting：luma 像素 (x,y) 属于色度块 (floor(x/2),
    // floor(y/2))。必须取该块 texel 中心（索引 +0.5）而不是 (src_x/2)+0.5：
    // 后者在奇数 luma 像素处落到两个色度 texel 的正中间，被
    // cudaFilterModeLinear 线性插值 50/50 混合 —— 与 CPU swscale 的
    // MPEG-2 siting（取所属 2x2 块的单个 texel）不一致，彩色边缘的
    // RGB 差可达 40+（实测 |Δ|>=8 像素 77-91% 落在奇数行/列）。
    // 修复后同帧 GPU/CPU 输出收敛到 ±3（仅舍入，实测 |Δ|>=8 = 0%）。
    auto uv = tex2D<float2>(chroma,
                            static_cast<int>(src_x * 0.5f) + 0.5f,
                            static_cast<int>(src_y * 0.5f) + 0.5f);
    yuv.u = uv.x;
    yuv.v = uv.y;

    if (output_format == 2) {
        // YUV420/NV12 packed 2D 输出：前 output_height 行是原始 Y
        // （调用方通过 get_color_range 自行做与 GRAY8 相同的展开），
        // 之后 output_height/2 行是原始 interleaved U/V。调用方
        // （VideoReader）保证输出宽高为偶数（ROI-first 路径按偶数超集
        // 重建输出池）。
        uint8_t* out = dst;
        out[dst_x + dst_y * output_width] =
            convert<uint8_t>(clip(yuv.y * 255.0f, 255.0f));
        if ((dst_x & 1) == 0 && (dst_y & 1) == 0) {
            uint8_t* uv_out = out
                + static_cast<size_t>(output_height) * output_width
                + static_cast<size_t>(dst_y / 2) * output_width
                + static_cast<size_t>(dst_x);
            uv_out[0] = convert<uint8_t>(clip(uv.x * 255.0f, 255.0f));
            uv_out[1] = convert<uint8_t>(clip(uv.y * 255.0f, 255.0f));
        }
        return;
    }

    T* out = dst + (dst_x + dst_y * output_width) * 3;
    yuv2rgb(yuv, out, 1, false);
}
}  // namespace detail

int DivUp(int total, int grain) {
    return (total + grain - 1) / grain;
}

void ProcessFrame(cudaTextureObject_t chroma, cudaTextureObject_t luma,
    uint8_t* dst, cudaStream_t stream, uint16_t input_width, uint16_t input_height,
    int output_width, int output_height,
    int src_x0, int src_y0, float fx, float fy, int bit_depth,
    int output_format, int color_range) {
    // resize factor: 0 表示由本函数按 in/out 推导（全帧缩放路径）；
    // ROI-first 路径由调用方传 1.0（窗口像素 1:1 映射）。
    // output_format: 0 = RGB24, 1 = GRAY8, 2 = YUV420/NV12 packed 2D
    // （Y 平面按 color_range 展开，U/V 原始）。
    if (fx <= 0.0f) fx = static_cast<float>(input_width) / output_width;
    if (fy <= 0.0f) fy = static_cast<float>(input_height) / output_height;
    // 位深语义：P016/P012 的 10/12-bit 数据在 16-bit 字中左对齐存储，
    // normalized float 采样 texel/65535 即等于 value/2^bit_depth —— 与
    // 8-bit 的 texel/255 语义天然一致，无需额外缩放（实测左对齐假设）。
    (void)bit_depth;

    dim3 block(32, 8);
    dim3 grid(DivUp(output_width, block.x), DivUp(output_height, block.y));

    detail::process_frame_kernel<<<grid, block, 0, stream>>>
            (luma, chroma, dst, input_width, input_height,
             output_width, output_height, src_x0, src_y0, fx, fy,
             output_format, color_range);
}
}  // namespace cuda
}  // namespace decord
