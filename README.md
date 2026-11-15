[简体中文](README.zh-CN.md)

# viewfeat

C++ library, version 0.1.0. It runs SuperPoint, ALIKED, LightGlue, CosPlace, EigenPlaces, and FlexGlue with TensorRT.

It does two things:

- Extract local features and match two images. A pipeline is one extractor plus one matcher: SuperPoint + LightGlue, ALIKED + LightGlue, SuperPoint + FlexGlue, or ALIKED + FlexGlue.
- Compute one global place descriptor with CosPlace or EigenPlaces. The vector is L2-normalized, so a dot product is the cosine similarity. `Create("default")` still selects CosPlace.

It does not estimate a camera pose. `Match(..., outlier_rejection=true)` only drops matches that fail a fundamental-matrix RANSAC (`cv::findFundamentalMat`, `FM_RANSAC`, threshold 3 px, confidence 0.99). Fewer than 8 matches, or a failed estimate, leaves the match list as it is.

The git tree has no weights and no engines. Build them into `weights/`. Sample images are in `example/image/`.

Public headers, installed with the library:

- `include/viewfeat/types.h` — `LocalFeatures`, `SetSearchPaths`
- `include/viewfeat/pipeline.h` — `FeaturePipeline`
- `include/viewfeat/retriever.h` — `GlobalRetriever`

Headers under `src/models/` and `src/runtime/` are private. They are not installed.

## Dependencies

Build and run need:

- CMake 3.16 or newer, a C++14 compiler
- CUDA
- TensorRT: `NvInfer.h`, `libnvinfer`, `libnvonnxparser`. CMake looks at `TENSORRT_ROOT`, then `/opt/TensorRT`, `/usr`, and `/usr/local`
- OpenCV, Eigen3, yaml-cpp

Inference needs an NVIDIA GPU. Loading an engine built for another GPU or another TensorRT version fails with `Failed to deserialize engine` and the text `TensorRT version or GPU architecture mismatch`.

`docker/Dockerfile` starts from `nvcr.io/nvidia/tensorrt:25.03-py3`. The file's comment says that base image is Ubuntu 24.04, Python 3.12, CUDA 12.8.1, and TensorRT 10.9, and that CUDA 12.8 is there for Blackwell GPUs such as the RTX 5070 Ti (compute capability 12.0). On top of that it installs CMake, Ninja, Eigen, OpenCV, yaml-cpp, and the Python export stack: `onnx`, `onnxruntime`, `onnxsim`, `torch==2.7.0+cu128`, `torchvision==0.22.0+cu128`. Apt and pip default to Aliyun mirrors. Override them with `--build-arg UBUNTU_MIRROR=...`, `PYPI_INDEX`, or `PYTORCH_WHEEL_INDEX`.

`package.xml` names the ROS 2 package `viewfeat` with `build_type` cmake and license Apache-2.0. There is no `LICENSE` file. TensorRT and CUDA are not ROS dependencies; install them on the system yourself. The package ships no nodes and no launch files.

## Build

On a machine that already has the C++ dependencies:

```bash
cmake -S . -B build
cmake --build build
```

The library target is `viewfeat` (`libviewfeat.so`). `3rdparty/tensorrtbuffer` is built as `libtensorrtbuffer.so` and linked privately. Examples are off unless you ask:

```bash
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build
```

Those binaries are `build/example/inference_image`, `inference_sequence`, `inference_cosplace`, and `inference_eigenplaces`. Each build copies `libviewfeat.so` and `libtensorrtbuffer.so` next to the executable and sets `RPATH` to `$ORIGIN`.

You can also configure from `example/`. That builds the same library under `example/build/viewfeat_core/` and the four programs under `example/build/`:

```bash
cmake -S example -B example/build
cmake --build example/build
```

If TensorRT is not in the default search path:

```bash
cmake -S . -B build -DTENSORRT_ROOT=/path/to/TensorRT
```

`TENSORRT_ROOT` may also be an environment variable when you run CMake.

Install:

```bash
cmake --install build
```

That installs the two shared libraries, the three public headers, `config/config.yaml` to `<prefix>/share/viewfeat/config/config.yaml`, and a CMake package `viewfeat::viewfeat` under `<prefix>/<libdir>/cmake/viewfeat`. `<libdir>` follows GNUInstallDirs (`lib` or `lib64`). If `weights/` exists at install time, `*.engine` and `*.onnx` in it are copied to `<prefix>/share/viewfeat/weights`. The default prefix is `/usr/local`.

In a ROS 2 workspace:

```bash
colcon build --packages-select viewfeat
```

Tests are on by default (`include(CTest)`). Pass `-DBUILD_TESTING=OFF` to skip them. See [Tests](#tests).

The container image used below does not get its tag from the Dockerfile. Build it yourself:

```bash
docker build -f docker/Dockerfile -t visual-localization-trt:blackwell-20261114 .
```

## Weights and engines

Create `weights/` at the repo root. `weights/`, `*.onnx`, `*.engine`, and `example/output/` are gitignored.

`config/config.yaml` names these files. Paths in the yaml are file names; the library joins them onto the model directory.

| Model | ONNX | Engine |
| --- | --- | --- |
| SuperPoint | `superpoint_v1.onnx` | `superpoint_v1.engine` |
| SuperPoint LightGlue | `superpoint_lightglue.onnx` | `superpoint_lightglue.engine` |
| ALIKED | `aliked-n16rot-top1024-512x288.onnx` | `aliked-n16rot-top1024-512x288.engine` |
| ALIKED LightGlue | `aliked_lightglue.onnx` | `aliked_lightglue.engine` |
| CosPlace | `cosplace.onnx` | `cosplace.engine` |
| EigenPlaces | `eigenplaces.onnx` | `eigenplaces.engine` |
| FlexGlue descriptor | `flexglue_descriptor.onnx` (export only) | `flexglue_descriptor.engine` |
| FlexGlue matcher | `flexglue_matcher.onnx` (export only) | `flexglue_matcher.engine` |

For SuperPoint, ALIKED, both LightGlue models, CosPlace, and EigenPlaces, startup loads the `.engine` if it deserializes. If it does not, the library builds an FP16 engine from the `.onnx` and writes the `.engine`. A missing ONNX prints `Failed to parse ONNX` and `Failed to build engine`.

FlexGlue only deserializes `flexglue_descriptor.engine` and `flexglue_matcher.engine`. There is no ONNX fallback. A missing file prints `Engine not found`.

Export SuperPoint, ALIKED, both LightGlue models, CosPlace, and EigenPlaces (this does not export FlexGlue):

```bash
./export_onnx.sh
```

That runs `python3 -m onnx_export export all` in `convert2onnx/`. Output goes to `weights/`. The first run downloads weights and needs network:

- SuperPoint: LightGlue release `superpoint_v1.pth`, cached as `weights/superpoint_v1.pth` if that path is missing
- LightGlue: `https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth` and `aliked_lightglue.pth`
- ALIKED: `https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16rot.pth`
- CosPlace: `torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet18", fc_output_dim=512)`
- EigenPlaces: `torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet18", fc_output_dim=512)`. That constructor also downloads a CosPlace hub checkpoint and then replaces it with `https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet18_512_eigenplaces.pth`. If the hub load fails, the exporter builds the same ResNet18 + GeM graph and loads that state dict directly.

Defaults match the yaml: 512×288, 1024 keypoints, ALIKED model `aliked-n16rot`, CosPlace input 320 and descriptor 512, EigenPlaces input 512 and descriptor 512. Shapes are fixed. Pass exporter flags through the script if you need to, for example `./export_onnx.sh --width 512 --height 288`. EigenPlaces does not take those width and height flags; its square side is `--input-size` on `python3 -m onnx_export export eigenplaces` (default 512).

FlexGlue is a separate step. The released matcher uses ONNX Runtime's `MultiHeadAttention`, which this script rewrites to standard ONNX before TensorRT can load it:

```bash
(cd convert2onnx && python3 -m onnx_export flexglue --output-dir ../weights)
```

Then build engines with `trtexec`. Missing ONNX files are skipped. FP16 is on by default. The FlexGlue matcher is built with a fixed 1×256×64×64 feature map and up to 1024 keypoints, which is what `config/config.yaml` says.

```bash
./build_trt.sh
./build_trt.sh --dry-run
./build_trt.sh --models superpoint aliked cosplace eigenplaces
```

`--models` accepts `superpoint`, `aliked`, `superpoint_lightglue`, `aliked_lightglue`, `cosplace`, `eigenplaces`, `flexglue_descriptor`, `flexglue_matcher`.

You can skip `./build_trt.sh` for the non-FlexGlue models and let the C++ library build the engine on first use. FlexGlue still needs the two engines from `trtexec`.

## Configuration

The file is `config/config.yaml`. It is read at runtime. Editing it does not require a rebuild. The in-process cache reloads the file when the path, the model directory, the mtime, or the size changes.

Search order for the config file and the model directory:

1. `FeaturePipeline::SetSearchPaths(config_path, model_dir)`, or `viewfeat::SetSearchPaths`. A non-empty path overrides. An empty string clears a previous override, and the search continues below. `GlobalRetriever` uses the same paths. The setting is process-wide.
2. Environment variables `VIEWFEAT_CONFIG_PATH` and `VIEWFEAT_MODEL_DIR`. An empty value is ignored.
3. The install paths baked in at configure time, if they exist: `<prefix>/share/viewfeat/config/config.yaml` and `<prefix>/share/viewfeat/weights`.
4. The source paths baked in at configure time, if they exist: `<source>/config/config.yaml` and `<source>/weights`.
5. If none of those exist, the library still tries the install path and fails.

A missing file prints `Config file ... does not exist` and `Create` fails with `Failed to build`. A file that is not a YAML map, or a section that is present but has a missing or mistyped key, makes `Create` return null and print `Failed to load config`. A section you leave out does not fail the whole file; that model is simply unavailable. `feature_type` is optional and defaults to `superpoint`.

`FeaturePipeline::Create` lowercases its argument.

| Argument | What is built |
| --- | --- |
| `""` or `"default"` | the `feature_type` value. A missing key still selects SuperPoint. The value `default` is not a family and fails. |
| `superpoint` | SuperPoint + SuperPoint LightGlue |
| `aliked` | ALIKED + ALIKED LightGlue |
| `superpoint_flexglue` | SuperPoint keypoints, FlexGlue dense features, FlexGlue matcher |
| `aliked_flexglue` | ALIKED keypoints, FlexGlue dense features, FlexGlue matcher |

Anything else prints `Unsupported feature family`. `GlobalRetriever::Create` accepts `""`, `"default"`, `"cosplace"`, and `"eigenplaces"`. `""` and `"default"` still select CosPlace. Anything else prints `Unsupported global retriever type`.

Images are `cv::Mat`. 1 channel is gray, 3 is BGR, 4 is BGRA. Other channel counts fail. SuperPoint is converted to gray. ALIKED, CosPlace, EigenPlaces, and FlexGlue convert to RGB. The pipeline resizes SuperPoint and ALIKED to `image_width` × `image_height` and scales keypoints back to the input image. CosPlace and EigenPlaces each resize to their own `input_size` × `input_size`. The FlexGlue descriptor resizes to its own `input_size`.

Knobs that the running code reads:

- `feature_type`: which family `Create("")` selects. The sample is `superpoint`.
- `superpoint.image_width`, `image_height`: sample `512` and `288`. The pipeline resizes to this, then the engine runs at this size. The default ONNX export is fixed at 512×288, so change the yaml and re-export together. `setInputShape` fails if they disagree.
- `superpoint.max_keypoints`, `keypoint_threshold` (`0.01`), `remove_borders` (`4`): applied in C++ after the network.
- `superpoint_lightglue.filter_threshold` (`0.1`). `num_keypoints` is optional; if you omit it, it copies `superpoint.max_keypoints`. The default LightGlue export is 1024 keypoints with a fixed shape. Keep this number equal to the exported engine.
- `aliked.image_width`, `image_height`: same 512×288 rule as SuperPoint. `descriptor_dim` must be `128` or `Create` returns null. The keypoint count used at runtime is the engine output shape. The default export name is `top1024`; changing `top_k` in the yaml alone does not change that shape.
- `aliked_lightglue.descriptor_dim` must be `128`. `filter_threshold` is `0.1` in the sample.
- `cosplace.input_size` is `320`, `descriptor_dim` is `512`. The engine's descriptor length must equal `descriptor_dim`.
- `eigenplaces.input_size` is `512`, `descriptor_dim` is `512`. Same tensor names as CosPlace (`images`, `descriptors`), different ONNX and engine files. The engine's descriptor length must equal `descriptor_dim`. The 512-pixel side matches the EigenPlaces training crop. Re-export if you change it; the GeM pooling kernel is fixed at export time.
- `flexglue`: `input_size` 512, `feature_dim` 256, `feature_width` and `feature_height` 64, `max_keypoints` 1024 (optional; the struct default is 1024), `filter_threshold` 0.1, `max_distance_px` 3, `min_covisibility` 0. `./build_trt.sh` builds the matcher for that feature-map size and at most 1024 keypoints. More keypoints than `max_keypoints` prints `Too many keypoints for the configured profile`. FlexGlue has no `onnx_file` and no `dla_core` keys.
- `dla_core` on the other models: a value `>= 0` turns on DLA when the library builds an engine from ONNX. The sample uses `-1`, which leaves DLA off. On a machine with no DLA cores, a core index prints an error and then hits `assert`.

Tensor name lists have to match the engine. The sample lists match the exporters in `convert2onnx/`. Wrong names fail with `Unexpected network IO count` or `Unexpected tensor layout`.

`LocalFeatures` holds `keypoints` in the input image frame, `descriptors` as N×D `CV_32F` (SuperPoint D=256, ALIKED D=128), `image_size`, and `family` (`"superpoint"` or `"aliked"`). It is move-only. `Match` rejects features that came from a different pipeline family.

## Run the examples

Build them with `-DBUILD_EXAMPLES=ON` or from the `example/` directory. Commands below assume the repo-root build. For the `example/build` tree, replace `build/example` with `example/build`.

`inference_image` matches two images. Usage:

```text
inference_image first_image_path second_image_path [output_image_path] [iterations] [superpoint|aliked|superpoint_flexglue|aliked_flexglue] [width] [height]
```

Defaults: output `match_image.png` in the current directory, 100 iterations, family `superpoint`, resize 512×288. The first iteration is not included in the printed averages. The picture is always labeled `SuperPoint LightGlue TensorRT`; the family you passed is printed on stdout.

```bash
mkdir -p example/output
./build/example/inference_image \
  example/image/image0.png example/image/image1.png \
  example/output/superpoint.png 10 superpoint
./build/example/inference_image \
  example/image/image0.png example/image/image1.png \
  example/output/aliked_flexglue.png 10 aliked_flexglue
```

`inference_sequence` matches each frame to the next one and writes `<i-1>_<i>.png`. The folder needs at least two images. The usage text calls the folder an absolute path; a relative path works.

```text
inference_sequence image_folder output_folder [superpoint|aliked|superpoint_flexglue|aliked_flexglue] [width] [height]
```

```bash
./build/example/inference_sequence \
  example/image/freiburg_sequence \
  example/output/sequence_superpoint \
  superpoint
```

`inference_cosplace` compares frame 0 with every later frame. It keeps `.jpg`, `.jpeg`, `.png`, and `.bmp`. Usage:

```text
inference_cosplace <image_sequence_dir> [--threshold <float>] [--output <csv>] [--max-images <int>]
```

Defaults: threshold `0.80`, CSV `cosplace_loop_similarity.csv` in the current directory, `--max-images 0` (no limit). Threshold must be in [-1, 1]. The CSV columns are `frame_index,image,similarity,is_loop`.

```bash
./build/example/inference_cosplace \
  example/image/freiburg_sequence \
  --threshold 0.80 \
  --output example/output/cosplace_loop_similarity.csv
```

`inference_eigenplaces` is the same comparison with `GlobalRetriever::Create("eigenplaces")`. Usage:

```text
inference_eigenplaces <image_sequence_dir> [--threshold <float>] [--output <csv>] [--max-images <int>]
```

Defaults: threshold `0.80`, CSV `eigenplaces_loop_similarity.csv` in the current directory, `--max-images 0`. The threshold is only the CSV cutoff you pass in. It is the same flag default as `inference_cosplace`, not a score measured for the EigenPlaces weights.

```bash
./build/example/inference_eigenplaces \
  example/image/freiburg_sequence \
  --threshold 0.80 \
  --output example/output/eigenplaces_loop_similarity.csv
```

Relative output paths follow the current directory. Config and `weights/` do not: they follow the search order above, so these binaries do not have to be started from the repo root.

`example/utils.h` is only for the samples (file listing and `cv::drawMatches`). It is not part of the library. More CMake snippets are in `example/LINKING.md`.

## Call it from your program

After `cmake --install`:

```cmake
find_package(viewfeat REQUIRED)
target_link_libraries(my_app PRIVATE viewfeat::viewfeat)
```

Or from the same build, without installing:

```cmake
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(/path/to/visual-localization-trt ${CMAKE_BINARY_DIR}/viewfeat_core)
target_link_libraries(my_app PRIVATE viewfeat::viewfeat)
```

```cpp
#include "viewfeat/pipeline.h"
#include "viewfeat/retriever.h"

viewfeat::SetSearchPaths("config/config.yaml", "weights");

auto pipeline = viewfeat::FeaturePipeline::Create("superpoint");
if (!pipeline) {
  return 1;
}
viewfeat::LocalFeatures features0;
viewfeat::LocalFeatures features1;
if (!pipeline->Extract(image0, features0) || !pipeline->Extract(image1, features1)) {
  return 1;
}
std::vector<cv::DMatch> matches;
if (!pipeline->Match(features0, features1, matches, false)) {
  return 1;
}

auto retriever = viewfeat::GlobalRetriever::Create("cosplace");
std::vector<float> descriptor;
if (!retriever || !retriever->Infer(image0, descriptor)) {
  return 1;
}

auto eigenplaces = viewfeat::GlobalRetriever::Create("eigenplaces");
```

`image0` and `image1` are `cv::Mat`. `descriptor.size()` equals `retriever->DescriptorDim()` (512 with the sample config, for both CosPlace and EigenPlaces). At run time the process still has to find `libviewfeat.so`, `libtensorrtbuffer.so`, and the TensorRT and CUDA libraries. The sample programs copy the first two beside the executable. An installed library's RPATH includes `$ORIGIN` and the TensorRT library directory found at build time.

Other `FeaturePipeline` methods that exist: `Match` with keypoints and descriptor matrices (LightGlue only; FlexGlue returns false because it needs the dense map), `MatchImages`, and `MatchWithImages` (FlexGlue, takes both images and the keypoints). `Family()` returns the pipeline or retriever name. `DescriptorDim()` is 256 for SuperPoint and 128 for ALIKED.

## Tests

Run the full ctest in `visual-localization-trt:blackwell-20261114` with the GPU visible. This is the command used on a host with an RTX 5070 Ti. `--gpus all` is required. Without it the container has no GPU: `engine_io` and `config_load` can still pass, and the five image tests exit 77, which ctest counts as skipped.

```bash
docker run --rm --gpus all -v "$PWD":/src -w /src visual-localization-trt:blackwell-20261114 \
  bash -lc 'nvidia-smi -L && cmake -S /src -B /tmp/viewfeat-build -G Ninja -DBUILD_TESTING=ON && cmake --build /tmp/viewfeat-build && ctest --test-dir /tmp/viewfeat-build --output-on-failure -j1'
```

`nvidia-smi -L` runs first and should list the GPU. The image tests also need the repo `weights/` directory (mounted at `/src/weights`) and these files, which are in the tree:

- `example/image/image0.png`
- `example/image/image1.png`
- `example/image/freiburg_sequence/1341847980.722988.png`
- `example/image/freiburg_sequence/1341847981.726650.png`

If `weights/` or one of those files is missing, the image test exits 77. If `weights/` exists but the engine or ONNX for that model does not load, the test fails. You do not need `-DBUILD_EXAMPLES=ON` for ctest. The `example_*` tests are `test/test_images.cpp`, not the example programs.

| ctest name | Command | GPU | Timeout |
| --- | --- | --- | --- |
| `engine_io` | `test_engine_io` | no | 60 s |
| `engine_io_garbage` | `test_engine_io --garbage` | deserialize path; exit 77 if there is no NVIDIA device and TensorRT throws or dies by a signal | 90 s |
| `config_load` | `test_config` | no | 60 s |
| `example_superpoint` | `test_images default` plus config, `weights/`, and the four images | yes, or exit 77 | 180 s |
| `example_aliked` | `test_images aliked` and the same paths | yes, or exit 77 | 180 s |
| `example_cosplace` | `test_images cosplace` and the same paths | yes, or exit 77 | 180 s |
| `example_eigenplaces` | `test_images eigenplaces` and the same paths | yes, or exit 77 | 300 s |
| `example_flexglue` | `test_images superpoint_flexglue` and the same paths | yes, or exit 77 | 300 s |

`example_flexglue` does not run `aliked_flexglue`. A device counts as present when `/dev/nvidia0`, `/dev/nvidiactl`, or `/dev/dxg` exists. The command above passes `-j1`.

Outside Docker, with the same dependencies installed and a visible GPU:

```bash
nvidia-smi -L
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure -j1
```

## Layout

```text
include/viewfeat/          public headers
src/                       pipeline, retriever, local features, models, runtime
config/config.yaml         sizes, thresholds, tensor names, weight file names
example/                   four programs, example/image, linking notes
example/image/             image0.png, image1.png, freiburg_sequence/
test/                      engine_io, config, and image tests
convert2onnx/              python -m onnx_export
export_onnx.sh             export SuperPoint, ALIKED, LightGlue, CosPlace, EigenPlaces into weights/
build_trt.sh               trtexec over ONNX files in weights/
docker/Dockerfile          image based on TensorRT 25.03
3rdparty/tensorrtbuffer/   TensorRT sample helpers, built as libtensorrtbuffer.so
cmake/                     CMake package template
package.xml                ROS 2 package name viewfeat
docs/                      internal review notes, not a user guide
```
