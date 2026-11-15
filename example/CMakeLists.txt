# Standalone build (recommended for trying examples):
#   cd example && cmake -B build -S . && cmake --build build -j
#
# Or from repo root:
#   cmake -B build -S . -DBUILD_EXAMPLES=ON && cmake --build build -j
#
# This file shows how to link viewfeat::viewfeat into your own targets.

cmake_minimum_required(VERSION 3.10)
project(viewfeat_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

get_filename_component(VIEWFEAT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(viewfeat_example_utils)

# --- Link viewfeat ------------------------------------------------------------
if(NOT TARGET viewfeat)
  # Standalone: build the core library from the parent project.
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory("${VIEWFEAT_ROOT}" "${CMAKE_BINARY_DIR}/viewfeat_core")
else()
  message(STATUS "Reusing existing viewfeat target from parent project")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d)

# --- Examples: viewfeat/pipeline.h or retriever.h + example/utils.h (visualization / file listing) ---
viewfeat_add_example(inference_image)
viewfeat_add_example(inference_sequence)
viewfeat_add_example(inference_cosplace)
viewfeat_add_example(inference_eigenplaces)

message(STATUS "viewfeat examples -> ${CMAKE_BINARY_DIR}")
message(STATUS "Run from repo root so config/weights resolve, e.g.:")
message(STATUS "  ./example/build/inference_image ../example/image/image0.png ../example/image/image1.png")
