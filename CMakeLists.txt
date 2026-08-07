cmake_minimum_required(VERSION 3.8.2)
project(decord C CXX)

# Utility functions
include(cmake/util/Util.cmake)
include(cmake/util/FindFFmpeg.cmake)
include(cmake/util/FindCUDA.cmake)
# include(cmake/util/FindCUDAArchFlags.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
decord_option(USE_CUDA "Build with CUDA" OFF)
decord_option(USE_MSVC_MT "Build with MT" OFF)

# Project
if(USE_CUDA)
project(decord C CXX CUDA)
endif(USE_CUDA)

# include directories
include_directories("include")
include_directories("3rdparty/dlpack/include")
include_directories("3rdparty/dmlc-core/include")

# initial variables
set(DECORD_LINKER_LIBS "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_policy(SET CMP0042 NEW)

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
  add_definitions(-DHalide_SHARED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  # Default to Release unless a build type was explicitly requested. With an
  # empty build type the old code path below used to inject "-O0 -g" into the
  # C++ flags, silently producing unoptimized binaries.
  if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No CMAKE_BUILD_TYPE specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
  else()
    # No -O flag here on purpose: CMake appends the per-config flags
    # (-O3 -DNDEBUG for Release) after these, which is where optimization
    # belongs. Hardcoding -O0 in this branch silently disabled optimization
    # for builds without a CMAKE_BUILD_TYPE.
    set(CMAKE_C_FLAGS "-Wall -fPIC -fvisibility=hidden ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
  endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "-Wno-unknown-pragmas ${CMAKE_CXX_FLAGS}")
  endif()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
  endif()
endif(MSVC)

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc" "src/*.cu")
FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB DECORD_CORE_SRCS src/*.cc src/runtime/*.cc src/video/*.cc src/sampler/*.cc src/audio/*.cc src/av_wrapper/*.cc)

# Module rules
include(cmake/modules/FFmpeg.cmake)
include(cmake/modules/CUDA.cmake)

# Targets

add_library(decord SHARED ${DECORD_CORE_SRCS} ${DECORD_FFMPEG_SRCS} ${NVDEC_SRCS} ${RUNTIME_CUDA_SRCS} ${NVDEC_CUDA_SRCS})

# Windows: copy the FFmpeg runtime DLLs next to the built decord.dll so the
# development build runs out of the box (libinfo.py loads decord.dll from
# build/Release, whose directory must contain the FFmpeg dependencies).
if(WIN32 AND FFMPEG_DIR)
  file(GLOB DECORD_FFMPEG_RUNTIME_DLLS "${FFMPEG_DIR}/bin/*.dll")
  if(DECORD_FFMPEG_RUNTIME_DLLS)
    add_custom_command(TARGET decord POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${DECORD_FFMPEG_RUNTIME_DLLS} $<TARGET_FILE_DIR:decord>)
  endif()
endif()

# target_compile_features(decord PUBLIC cxx_std_11)

target_link_libraries(decord ${DECORD_LINKER_LIBS} ${DECORD_RUNTIME_LINKER_LIBS})
set_property(TARGET decord PROPERTY CUDA_STANDARD 11)

# More target definitions
if(MSVC)
  target_compile_definitions(decord PRIVATE -DDECORD_EXPORTS)
endif()

# Tests
set(TEST_EXECS "")
file(GLOB_RECURSE TEST_SRCS tests/cpp/*.cc)
find_library(GTEST_LIB gtest "$ENV{GTEST_LIB}")

if(GTEST_LIB)
  foreach(__srcpath ${TEST_SRCS})
    get_filename_component(__srcname ${__srcpath} NAME)
    string(REPLACE ".cc" "" __execname ${__srcname})
    add_executable(${__execname} ${__srcpath})
    list(APPEND TEST_EXECS ${__execname})
    if (MSVC)
      set(GTEST_ADD_LIBS "")
    else(MSVC)
      set(GTEST_ADD_LIBS pthread dl)
    endif(MSVC)
    target_link_libraries(${__execname}
        decord ${GTEST_LIB} ${GTEST_ADD_LIBS})
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_ALL 1)
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  endforeach()
  add_custom_target(cpptest DEPENDS ${TEST_EXECS})
  list(LENGTH TEST_EXECS NUM_TESTS)
  message(STATUS "Build with: " ${NUM_TESTS}  " tests")
endif()

# Installation rules
# The library installs into the python package directory by default so that
# wheels built with scikit-build-core (see pyproject.toml) and plain
# `cmake --install` both place libdecord next to the python code, where
# decord/_ffi/libinfo.py searches for it.  Override the destination with
# -DDECORD_INSTALL_LIBDIR=... if a plain library install is wanted.
if(NOT DEFINED DECORD_INSTALL_LIBDIR)
  set(DECORD_INSTALL_LIBDIR decord)
endif()
install(TARGETS decord
  LIBRARY DESTINATION ${DECORD_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${DECORD_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${DECORD_INSTALL_LIBDIR})
