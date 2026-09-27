#--------------------------------------------------------------------
#  Template custom cmake configuration for compiling
#
#  This file is used to override the build options in build.
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ mkdir build
#  $ cp cmake/config.cmake build
#
#  Next modify the according entries, and then compile by
#
#  $ cd build
#  $ cmake ..
#
#  Then buld in parallel with 8 threads
#
#  $ make -j8
#--------------------------------------------------------------------

#---------------------------------------------
# Backend runtimes.
#---------------------------------------------

# Whether enable CUDA during compile,
#
# 2026-09-28 R3-2b GPU-only 化：默认 ON 且不可关（CMakeLists 对显式
# OFF 直接 FATAL_ERROR）。本 fork 的 CUDA 构建零 Toolkit 依赖（驱动
# API 动态加载），无 GPU 机器运行时优雅降级 CPU 解码——上游原版的
# 「无 CUDA 构建物」需求由上游满足。
set(USE_CUDA ON)

# Whether build with MT mode on Windows
#
# Possible values:
# - ON: enable MT
# - OFF: disalbe MT
set(USE_MSVC_MT OFF)