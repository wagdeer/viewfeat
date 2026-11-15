# Linking `viewfeat` in your project

Public API (install / include path):

```text
include/viewfeat/types.h
include/viewfeat/pipeline.h
include/viewfeat/retriever.h
```

Do not include headers under `src/models/` or `src/runtime/` — they are compiled into `libviewfeat.so` and are not installed.

Library target: `viewfeat` (exported / alias: `viewfeat::viewfeat`)

Config and TensorRT engine paths are resolved at runtime, in order:

1. `FeaturePipeline::SetSearchPaths(config_path, model_dir)` if you call it (`GlobalRetriever` reads the same paths)
2. `VIEWFEAT_CONFIG_PATH` / `VIEWFEAT_MODEL_DIR` environment variables
3. Installed `share/viewfeat/config/config.yaml` and `share/viewfeat/weights/`
4. Source-tree `config/` and `weights/` (so examples still run from the repo root)

Your app only needs headers + the shared library (+ OpenCV if you read/write images).

## Option A — `find_package` (colcon / installed)

After `colcon build --packages-select viewfeat` (or `cmake --install`):

```cmake
find_package(viewfeat REQUIRED)
target_link_libraries(my_app PRIVATE viewfeat::viewfeat)
```

## Option B — `add_subdirectory` (same CMake build)

```cmake
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(/path/to/viewfeat ${CMAKE_BINARY_DIR}/viewfeat_core)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE viewfeat::viewfeat)
# viewfeat::viewfeat already exports include/viewfeat via PUBLIC include dirs
```

`main.cpp`:

```cpp
#include "viewfeat/pipeline.h"

auto pipeline = viewfeat::FeaturePipeline::Create("superpoint");
if (!pipeline) {
  return 1;
}
viewfeat::LocalFeatures features0;
viewfeat::LocalFeatures features1;
pipeline->Extract(image0, features0);
pipeline->Extract(image1, features1);
pipeline->Match(features0, features1, matches);
```

Runtime: ensure `libviewfeat.so` (and `libtensorrtbuffer.so` if not on `LD_LIBRARY_PATH`) are found. The examples copy both next to the executable and set `RPATH=$ORIGIN` — see `cmake/viewfeat_example_utils.cmake`.

## Option C — prebuilt library (manual)

After you build/install `viewfeat`:

```cmake
set(VIEWFEAT_ROOT "/path/to/viewfeat")
find_library(VIEWFEAT_LIBRARY NAMES viewfeat
  PATHS "${VIEWFEAT_ROOT}/build" "${VIEWFEAT_ROOT}/example/build/viewfeat_core" NO_DEFAULT_PATH)
find_path(VIEWFEAT_INCLUDE_DIR NAMES viewfeat/pipeline.h
  PATHS "${VIEWFEAT_ROOT}/include" NO_DEFAULT_PATH)

add_library(viewfeat::viewfeat SHARED IMPORTED)
set_target_properties(viewfeat::viewfeat PROPERTIES
  IMPORTED_LOCATION "${VIEWFEAT_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${VIEWFEAT_INCLUDE_DIR}"
)

target_link_libraries(my_app PRIVATE viewfeat::viewfeat)
```

Run with:

```bash
export LD_LIBRARY_PATH=/path/to/dir/with/libviewfeat.so:$LD_LIBRARY_PATH
./my_app
```

Or deploy `libviewfeat.so` beside your binary and use `RPATH=$ORIGIN` (same as the example helpers).

## Build the shipped examples

```bash
cd example
cmake -B build -S .
cmake --build build -j$(nproc)
```

Binaries: `example/build/inference_*`  
Shared libs copied next to them: `libviewfeat.so`, `libtensorrtbuffer.so`

Run from **repository root** (source-tree `config/` and `weights/` are the fallback search paths):

```bash
./example/build/inference_image example/image/image0.png example/image/image1.png
```

## Dependencies (transitive)

`viewfeat` itself needs TensorRT, CUDA, OpenCV, Eigen3, yaml-cpp, and `tensorrtbuffer` at link time. Your executable must link `viewfeat::viewfeat` only; TensorRT/OpenCV are already linked into the `.so`. At runtime you still need those system libraries on `LD_LIBRARY_PATH` / default paths (same as running the examples).
