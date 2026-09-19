/*!
 * \file nvml_min.h
 * \brief NVML 最小垫片（2026-09-19 冻结前瘦身）。
 *
 * 替代 14,546 行的官方 nvml.h 副本——本 fork 只用到 4 个符号，且**全部
 * 只服务一个判断**：驱动内核模块版本是否 ≥ 384（决定用不用自建 stream，
 * 见 cuda_threaded_decoder.cc:38-61）。
 *
 * 取值与官方头一致（ABI 稳定，NVML 自 2012 起未变）：
 *   nvmlReturn_t = unsigned int；NVML_SUCCESS = 0；
 *   NVML_ERROR_LIBRARY_NOT_FOUND = 4；
 *   NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE = 80。
 *
 * ⚠️ 函数指针的**真实解析**在 nv_gpu_dyn.cc 的 NV_NVML_FUNCS 表（运行时
 * 从 nvml.dll / libnvidia-ml.so.1 取符号），本头只提供类型与常量。
 * 若将来需要更多 NVML 能力（温度/时钟/功耗），应恢复官方头而非在此堆叠。
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/** NVML 返回码（官方 nvmlReturn_t 为 unsigned int，值域见官方头） */
typedef unsigned int nvmlReturn_t;

/** 成功 */
#define NVML_SUCCESS 0u
/** 驱动库未找到（本 fork 用作"未加载 NVML"的哨兵值） */
#define NVML_ERROR_LIBRARY_NOT_FOUND 4u

/** 驱动版本字符串缓冲长度（官方值 80） */
#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE 80

/** 初始化 NVML（真实符号由 nv_gpu_dyn 动态解析后调用） */
nvmlReturn_t nvmlInit(void);

/** 取驱动版本字符串（真实符号由 nv_gpu_dyn 动态解析后调用） */
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length);

#ifdef __cplusplus
}
#endif
