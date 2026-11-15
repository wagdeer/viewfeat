# Examples

Four programs demonstrate the public API in `include/viewfeat/types.h`, `pipeline.h`, and `retriever.h` (`FeaturePipeline`, `GlobalRetriever`). They do **not** include model headers.

## Build (recommended: from this directory)

```bash
cd example
cmake -B build -S .
cmake --build build -j$(nproc)
```

Artifacts:

| Output | Location |
|--------|----------|
| Executables | `example/build/inference_*` |
| Core library (build tree) | `example/build/viewfeat_core/libviewfeat.so` |
| Deployed next to each exe | `libviewfeat.so`, `libtensorrtbuffer.so` |

`RPATH=$ORIGIN` and the copied `.so` files avoid picking up an unrelated `libviewfeat.so` from the environment.

Alternative from repo root:

```bash
cmake -B build -S . -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
# binaries under build/ (no automatic .so copy; prefer example/ build above)
```

## Prerequisites

1. Export ONNX and run `./build_trt.sh` (see [project README](../README.md)); `weights/*.engine` must exist.
2. Run inference commands from the **repository root** so source-tree `config/` and `weights/` resolve (or set `VIEWFEAT_CONFIG_PATH` / `VIEWFEAT_MODEL_DIR`).

## How to link in your CMake project

See [LINKING.md](LINKING.md). Minimal pattern after install:

```cmake
find_package(viewfeat REQUIRED)
target_link_libraries(my_app PRIVATE viewfeat::viewfeat)
```

Implementation reference: `cmake/viewfeat_example_utils.cmake` (`viewfeat_add_example`).

## Bundled data

```text
include/viewfeat/types.h       # public feature types
include/viewfeat/pipeline.h    # FeaturePipeline
include/viewfeat/retriever.h   # GlobalRetriever

src/models/                    # model implementation (not installed)
src/runtime/                   # config and search paths (not installed)
example/utils.h                # demo helpers only (not part of the library)
```

## 1. `inference_image` — match two images

```bash
./example/build/inference_image \
  example/image/image0.png example/image/image1.png \
  example/image/superpoint_match.png 2 superpoint
```

Arguments: `image0` `image1` `[output]` `[iterations]` `[superpoint|aliked|superpoint_flexglue|aliked_flexglue]` `[width]` `[height]`. The FlexGlue variants require the optional FlexGlue engines described in the root README.  
Defaults: iterations `100`, family `superpoint`, size `512x288` (match `config/config.yaml`).

## 2. `inference_sequence` — adjacent frames

```bash
mkdir -p example/output/sequence_superpoint
./example/build/inference_sequence \
  example/image/freiburg_sequence \
  example/output/sequence_superpoint \
  superpoint
```

Output: `0_1.png`, `1_2.png`, …

## 3. `inference_cosplace` — global similarity

```bash
./example/build/inference_cosplace \
  example/image/freiburg_sequence \
  --threshold 0.80 --output example/output/cosplace_loop_similarity.csv
```

## 4. `inference_eigenplaces` — global similarity

Same loop comparison as `inference_cosplace`, using `GlobalRetriever::Create("eigenplaces")`. Input is 512×512. The `--threshold` default `0.80` is the CSV cutoff, not a score measured on these weights.

```bash
./example/build/inference_eigenplaces \
  example/image/freiburg_sequence \
  --threshold 0.80 --output example/output/eigenplaces_loop_similarity.csv
```

## Notes

- Rebuild the library after changing `config/config.yaml` or engines (`cmake --build example/build` rebuilds `viewfeat_core` automatically).
- Override config/weights at library build time via `VIEWFEAT_CONFIG_PATH` / `VIEWFEAT_MODEL_DIR` in the root `CMakeLists.txt`.
