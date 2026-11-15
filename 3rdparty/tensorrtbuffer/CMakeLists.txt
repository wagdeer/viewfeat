cmake_minimum_required(VERSION 3.10)
project(tensorrtbuffer)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB)
    message(FATAL_ERROR "TensorRT variables must be set by the parent CMakeLists.txt")
endif()

add_library(tensorrtbuffer SHARED
  src/logger.cpp
  src/sampleOptions.cpp
  src/sampleUtils.cpp
  src/bfloat16.cpp
  src/utils/timingCache.cpp
  src/utils/fileLock.cpp
)

target_include_directories(tensorrtbuffer
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/utils
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(tensorrtbuffer
  PRIVATE
    ${NVINFER_LIB}
    ${CUDA_LIBRARIES}
)
