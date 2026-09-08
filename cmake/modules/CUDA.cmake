# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# CUDA Module（0.8.1 去 Toolkit 化）
#
# 构建不再需要 CUDA Toolkit（无 nvcc、无 cudart、无 toolkit 头文件）：
# - 驱动 API / NVCUVID / NVML 运行时动态加载（nv_gpu_dyn.cc，导入表无
#   驱动 DLL —— 无 NVIDIA 驱动设备可正常加载并回退 CPU 解码）
# - cudart 调用走自研垫片（src/runtime/cuda/cudart_shim.*，转驱动 API）
# - improc kernel 以预编译 PTX 内嵌（src/improc/improc_ptx.inc，经
#   cuModuleLoadData 加载；再生成命令见 improc_dyn.cc 头注释）
# - nvcuvid.h / cuda.h / nvml.h 头文件由仓库自带（cuda_include/、nvcuvid/）

if(USE_CUDA)
  add_definitions(-DDECORD_USE_CUDA)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/video/nvcodec/cuda_include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/video/nvcodec/nvcuvid)
  message(STATUS "Build with CUDA support (driver APIs dynamically loaded, no CUDA Toolkit required)")
  file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
  file(GLOB NVDEC_SRCS src/video/nvcodec/*.cc)
  file(GLOB NVDEC_CC_SRCS src/improc/*.cc)
  list(APPEND NVDEC_SRCS ${NVDEC_CC_SRCS})
  set(NVDEC_CUDA_SRCS "")  # 不再用 nvcc（kernel 以内嵌 PTX 提供）
else(USE_CUDA)
  message(STATUS "CUDA disabled, no nvdec capabilities will be enabled...")
  set(NVDEC_SRCS "")
  set(RUNTIME_CUDA_SRCS "")
  set(NVDEC_CUDA_SRCS "")
endif(USE_CUDA)
