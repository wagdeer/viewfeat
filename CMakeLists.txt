cmake_minimum_required(VERSION 3.16)
project(viewfeat VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Keep configuration minimal: allow one optional override and auto-discover elsewhere.
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "Optional TensorRT install root")
set(_TENSORRT_HINTS)
if(TENSORRT_ROOT)
  list(APPEND _TENSORRT_HINTS ${TENSORRT_ROOT})
endif()
list(APPEND _TENSORRT_HINTS /opt/TensorRT /usr /usr/local)

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${_TENSORRT_HINTS}
  PATH_SUFFIXES include
)

find_library(NVINFER_LIB
  NAMES nvinfer
  HINTS ${_TENSORRT_HINTS}
  PATH_SUFFIXES lib lib64 targets/aarch64-linux-gnu/lib lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

find_library(NVONNXPARSER_LIB
  NAMES nvonnxparser
  HINTS ${_TENSORRT_HINTS}
  PATH_SUFFIXES lib lib64 targets/aarch64-linux-gnu/lib lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
  message(FATAL_ERROR
    "TensorRT not found. Set TENSORRT_ROOT or pass -DTENSORRT_ROOT=/path/to/TensorRT")
endif()

get_filename_component(TENSORRT_LIB_DIR "${NVINFER_LIB}" DIRECTORY)

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${NVINFER_LIB}")
message(STATUS "TensorRT nvonnxparser: ${NVONNXPARSER_LIB}")

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(viewfeat SHARED
  include/viewfeat/types.h
  include/viewfeat/pipeline.h
  include/viewfeat/retriever.h
  src/models/superpoint.cpp
  src/models/superpoint_lightglue.cpp
  src/models/aliked.cpp
  src/models/aliked_lightglue.cpp
  src/models/place_descriptor.cpp
  src/models/cosplace.cpp
  src/models/eigenplaces.cpp
  src/models/flexglue.cpp
  src/runtime/config.cpp
  src/runtime/engine_io.cpp
  src/runtime/search_paths.cpp
  src/local_features.cpp
  src/pipeline.cpp
  src/retriever.cpp
)

set_target_properties(viewfeat PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${TENSORRT_LIB_DIR}"
  INSTALL_RPATH "$ORIGIN;${TENSORRT_LIB_DIR}"
)

set_target_properties(tensorrtbuffer PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${TENSORRT_LIB_DIR}"
  INSTALL_RPATH "$ORIGIN;${TENSORRT_LIB_DIR}"
)

# PUBLIC: include/viewfeat/{types,pipeline,retriever}.h
# PRIVATE: model and runtime headers live under src/ and are not installed
target_include_directories(viewfeat
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer/include
    ${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer/include/utils
    ${TENSORRT_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

set(VIEWFEAT_INSTALL_CONFIG_PATH
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/viewfeat/config/config.yaml")
set(VIEWFEAT_INSTALL_MODEL_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/viewfeat/weights")

target_compile_definitions(viewfeat
  PRIVATE
    VIEWFEAT_CONFIG_PATH="${VIEWFEAT_INSTALL_CONFIG_PATH}"
    VIEWFEAT_MODEL_DIR="${VIEWFEAT_INSTALL_MODEL_DIR}"
    VIEWFEAT_SOURCE_CONFIG_PATH="${PROJECT_SOURCE_DIR}/config/config.yaml"
    VIEWFEAT_SOURCE_MODEL_DIR="${PROJECT_SOURCE_DIR}/weights"
)

target_compile_features(viewfeat PUBLIC cxx_std_14)

target_link_libraries(viewfeat
  PRIVATE
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${CUDA_LIBRARIES}
    tensorrtbuffer
)

if(TARGET OpenCV::core)
  target_link_libraries(viewfeat PUBLIC OpenCV::core)
else()
  target_include_directories(viewfeat PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(viewfeat PRIVATE ${OpenCV_LIBS})

if(TARGET Eigen3::Eigen)
  target_link_libraries(viewfeat PRIVATE Eigen3::Eigen)
endif()

if(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(viewfeat PRIVATE yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  target_link_libraries(viewfeat PRIVATE yaml-cpp)
else()
  target_link_libraries(viewfeat PRIVATE ${YAML_CPP_LIBRARIES})
endif()

add_library(viewfeat::viewfeat ALIAS viewfeat)

option(BUILD_EXAMPLES "Build example/ programs (see example/README.md)" OFF)

if(BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

install(
  TARGETS viewfeat tensorrtbuffer
  EXPORT viewfeatTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  FILES config/config.yaml
  DESTINATION ${CMAKE_INSTALL_DATADIR}/viewfeat/config
)

if(EXISTS "${PROJECT_SOURCE_DIR}/weights")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/weights/"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/viewfeat/weights
    FILES_MATCHING
      PATTERN "*.engine"
      PATTERN "*.onnx"
  )
endif()

install(
  EXPORT viewfeatTargets
  FILE viewfeatTargets.cmake
  NAMESPACE viewfeat::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/viewfeat
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/viewfeatConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/viewfeatConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/viewfeatConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/viewfeat
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/viewfeatConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/viewfeatConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/viewfeat
)
