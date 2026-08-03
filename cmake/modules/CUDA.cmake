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

# CUDA Module
find_cuda(${USE_CUDA})

if(CUDA_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
  include_directories(${CUDA_INCLUDE_DIRS})
  add_definitions(-DDECORD_USE_CUDA)
endif(CUDA_FOUND)

if(USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif()
  # 不再链接 nvml.lib / nvcuvid.lib（Video Codec SDK）：
  # CUDA 驱动 API / NVCUVID / NVML 全部改为运行时动态加载（nv_gpu_dyn.cc），
  # decord.dll 导入表不再依赖驱动 DLL → 无 NVIDIA 驱动设备可正常加载并
  # 回退 CPU 解码。nvcuvid.h 头文件由仓库自带（src/video/nvcodec/nvcuvid/）。
  message(STATUS "Build with CUDA support (GPU APIs dynamically loaded)")
  file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
  file(GLOB NVDEC_SRCS src/video/nvcodec/*.cc)
  file(GLOB NVDEC_CUDA_SRCS src/improc/*.cu)

  list(APPEND DECORD_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
  list(APPEND DECORD_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND DECORD_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND DECORD_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})

else(USE_CUDA)
  message(STATUS "CUDA disabled, no nvdec capabilities will be enabled...")
  set(NVDEC_SRCS "")
  set(RUNTIME_CUDA_SRCS "")
endif(USE_CUDA)
