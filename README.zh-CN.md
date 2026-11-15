[English](README.md)

# viewfeat

C++ 库，版本 0.1.0。用 TensorRT 跑 SuperPoint、ALIKED、LightGlue、CosPlace、EigenPlaces 和 FlexGlue。

库做两件事：

- 提局部特征，并匹配两张图。一条流水线是一个提取器加一个匹配器：SuperPoint + LightGlue、ALIKED + LightGlue、SuperPoint + FlexGlue，或 ALIKED + FlexGlue。
- 用 CosPlace 或 EigenPlaces 算一个全局地点描述子。向量做了 L2 归一化，点积就是余弦相似度。`Create("default")` 仍然选 CosPlace。

库不估计相机位姿。`Match(..., outlier_rejection=true)` 只用基础矩阵 RANSAC 丢掉外点（`cv::findFundamentalMat`，`FM_RANSAC`，阈值 3 像素，置信度 0.99）。匹配少于 8 个，或者估计失败，匹配列表保持原样。

git 里没有权重，也没有 engine。自己生成，放到 `weights/`。示例图片在 `example/image/`。

随库安装的公开头文件：

- `include/viewfeat/types.h`：`LocalFeatures`、`SetSearchPaths`
- `include/viewfeat/pipeline.h`：`FeaturePipeline`
- `include/viewfeat/retriever.h`：`GlobalRetriever`

`src/models/` 和 `src/runtime/` 下的头文件是内部的，不安装。

## 依赖

编译和运行需要：

- CMake 3.16 或更新，C++14 编译器
- CUDA
- TensorRT：`NvInfer.h`、`libnvinfer`、`libnvonnxparser`。CMake 先看 `TENSORRT_ROOT`，再看 `/opt/TensorRT`、`/usr`、`/usr/local`
- OpenCV、Eigen3、yaml-cpp

推理需要 NVIDIA GPU。换了一块 GPU 或另一个 TensorRT 版本编出来的 engine，加载时会失败，日志是 `Failed to deserialize engine`，并带有 `TensorRT version or GPU architecture mismatch`。

`docker/Dockerfile` 的基础镜像是 `nvcr.io/nvidia/tensorrt:25.03-py3`。文件里的注释写明：这个基础镜像是 Ubuntu 24.04、Python 3.12、CUDA 12.8.1、TensorRT 10.9；CUDA 12.8 是给 Blackwell GPU 用的，例如 RTX 5070 Ti（计算能力 12.0）。镜像里另外装了 CMake、Ninja、Eigen、OpenCV、yaml-cpp，以及导出用的 Python 包：`onnx`、`onnxruntime`、`onnxsim`、`torch==2.7.0+cu128`、`torchvision==0.22.0+cu128`。apt 和 pip 默认走阿里云镜像，可以用 `--build-arg UBUNTU_MIRROR=...`、`PYPI_INDEX`、`PYTORCH_WHEEL_INDEX` 换掉。

`package.xml` 里的 ROS 2 包名是 `viewfeat`，`build_type` 是 cmake，license 写的是 Apache-2.0。仓库里没有 `LICENSE` 文件。TensorRT 和 CUDA 不在这份 ROS 依赖里，要自己装到系统上。这个包没有节点，也没有 launch 文件。

## 构建

机器上已经有上面这些 C++ 依赖时：

```bash
cmake -S . -B build
cmake --build build
```

库目标是 `viewfeat`（`libviewfeat.so`）。`3rdparty/tensorrtbuffer` 编成 `libtensorrtbuffer.so`，作为私有依赖链接。示例默认不编，要打开开关：

```bash
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build
```

四个程序在 `build/example/inference_image`、`inference_sequence`、`inference_cosplace`、`inference_eigenplaces`。构建时会把 `libviewfeat.so` 和 `libtensorrtbuffer.so` 拷到可执行文件旁边，并把 `RPATH` 设成 `$ORIGIN`。

也可以在 `example/` 里单独配置。库在 `example/build/viewfeat_core/`，四个程序在 `example/build/`：

```bash
cmake -S example -B example/build
cmake --build example/build
```

TensorRT 不在默认搜索路径里时：

```bash
cmake -S . -B build -DTENSORRT_ROOT=/path/to/TensorRT
```

跑 CMake 时也可以用环境变量 `TENSORRT_ROOT`。

安装：

```bash
cmake --install build
```

装上去的是两个动态库、三个公开头文件、`config/config.yaml`（到 `<prefix>/share/viewfeat/config/config.yaml`），以及 CMake 包 `viewfeat::viewfeat`（在 `<prefix>/<libdir>/cmake/viewfeat`）。`<libdir>` 按 GNUInstallDirs，一般是 `lib` 或 `lib64`。安装时如果存在 `weights/`，其中的 `*.engine` 和 `*.onnx` 会拷到 `<prefix>/share/viewfeat/weights`。默认前缀是 `/usr/local`。

放在 ROS 2 工作空间里时：

```bash
colcon build --packages-select viewfeat
```

测试默认会编（`include(CTest)`）。不想编测试就加 `-DBUILD_TESTING=OFF`。怎么跑见 [测试](#测试)。

下面用到的镜像标签不是 Dockerfile 写死的，需要自己打：

```bash
docker build -f docker/Dockerfile -t visual-localization-trt:blackwell-20261114 .
```

## 权重和 engine

在仓库根目录建 `weights/`。`weights/`、`*.onnx`、`*.engine`、`example/output/` 都在 `.gitignore` 里。

文件名以 `config/config.yaml` 为准。yaml 里写的是文件名，库会把它拼到模型目录后面。

| 模型 | ONNX | Engine |
| --- | --- | --- |
| SuperPoint | `superpoint_v1.onnx` | `superpoint_v1.engine` |
| SuperPoint LightGlue | `superpoint_lightglue.onnx` | `superpoint_lightglue.engine` |
| ALIKED | `aliked-n16rot-top1024-512x288.onnx` | `aliked-n16rot-top1024-512x288.engine` |
| ALIKED LightGlue | `aliked_lightglue.onnx` | `aliked_lightglue.engine` |
| CosPlace | `cosplace.onnx` | `cosplace.engine` |
| EigenPlaces | `eigenplaces.onnx` | `eigenplaces.engine` |
| FlexGlue 描述子 | `flexglue_descriptor.onnx`（只在导出时用） | `flexglue_descriptor.engine` |
| FlexGlue 匹配器 | `flexglue_matcher.onnx`（只在导出时用） | `flexglue_matcher.engine` |

SuperPoint、ALIKED、两份 LightGlue、CosPlace、EigenPlaces 启动时先加载 `.engine`。反序列化失败就用 `.onnx` 编一个 FP16 engine，并写成 `.engine`。ONNX 不在会打印 `Failed to parse ONNX` 和 `Failed to build engine`。

FlexGlue 只反序列化 `flexglue_descriptor.engine` 和 `flexglue_matcher.engine`，不会从 ONNX 现编。文件不在时打印 `Engine not found`。

这个脚本导出 SuperPoint、ALIKED、两份 LightGlue、CosPlace 和 EigenPlaces。它不导出 FlexGlue：

```bash
./export_onnx.sh
```

脚本在 `convert2onnx/` 里执行 `python3 -m onnx_export export all`，输出到 `weights/`。第一次会下载权重，需要网络：

- SuperPoint：LightGlue 发布的 `superpoint_v1.pth`。`weights/superpoint_v1.pth` 不存在时会下载并缓存到这个路径
- LightGlue：`https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth` 和 `aliked_lightglue.pth`
- ALIKED：`https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16rot.pth`
- CosPlace：`torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet18", fc_output_dim=512)`
- EigenPlaces：`torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet18", fc_output_dim=512)`。这个构造函数会先下载一份 CosPlace hub 权重，再用 `https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet18_512_eigenplaces.pth` 覆盖。hub 加载失败时，导出器按同一套 ResNet18 + GeM 图自己建网，并直接加载这份 state dict。

默认尺寸和 yaml 一致：512×288，1024 个点，ALIKED 模型名 `aliked-n16rot`，CosPlace 输入 320、描述子 512，EigenPlaces 输入 512、描述子 512。形状是固定的。导出参数可以跟着脚本传，例如 `./export_onnx.sh --width 512 --height 288`。EigenPlaces 不吃这对宽高，正方形边长在 `python3 -m onnx_export export eigenplaces --input-size` 上，默认 512。

FlexGlue 要单独做。上游匹配器使用 ONNX Runtime 的 `MultiHeadAttention`，这个脚本会把它改写成标准 ONNX，TensorRT 才能加载：

```bash
(cd convert2onnx && python3 -m onnx_export flexglue --output-dir ../weights)
```

再用 `trtexec` 编 engine。缺的 ONNX 会跳过。默认开 FP16。FlexGlue 匹配器按 1×256×64×64 的特征图、最多 1024 个点来编，和 `config/config.yaml` 一致。

```bash
./build_trt.sh
./build_trt.sh --dry-run
./build_trt.sh --models superpoint aliked cosplace eigenplaces
```

`--models` 可以是 `superpoint`、`aliked`、`superpoint_lightglue`、`aliked_lightglue`、`cosplace`、`eigenplaces`、`flexglue_descriptor`、`flexglue_matcher`。

除了 FlexGlue，也可以不跑 `./build_trt.sh`，等 C++ 库第一次用到时自己从 ONNX 编。FlexGlue 仍然要先有 `trtexec` 编出的两个 engine。

## 配置

配置文件是 `config/config.yaml`，运行时读取。改它不用重新编译。进程内缓存会在路径、模型目录、mtime 或文件大小变化后重新加载。

配置文件和模型目录按这个顺序找：

1. `FeaturePipeline::SetSearchPaths(config_path, model_dir)`，或 `viewfeat::SetSearchPaths`。非空路径会覆盖。空字符串会清掉上一次的覆盖，然后继续按下面的顺序找。`GlobalRetriever` 读同一组路径。这个设置对整个进程有效。
2. 环境变量 `VIEWFEAT_CONFIG_PATH` 和 `VIEWFEAT_MODEL_DIR`。空值忽略。
3. 配置 CMake 时写进库里的安装路径。运行时这个文件或目录存在才用：`<prefix>/share/viewfeat/config/config.yaml` 和 `<prefix>/share/viewfeat/weights`。
4. 配置 CMake 时写进库里的源码路径。运行时这个文件或目录存在才用：`<source>/config/config.yaml` 和 `<source>/weights`。
5. 上面都没有，库仍会去试安装路径，然后失败。

文件不存在时打印 `Config file ... does not exist`，`Create` 以 `Failed to build` 失败。文件不是 YAML map，或者某个写出的段缺键、类型不对，`Create` 返回空指针，并打印 `Failed to load config`。某一整段不写，文件仍然有效，只是那个模型建不起来。`feature_type` 可以不写，默认 `superpoint`。

`FeaturePipeline::Create` 会把参数转成小写。

| 参数 | 建出来的东西 |
| --- | --- |
| `""` 或 `"default"` | 用 `feature_type`。没有这个键时仍是 SuperPoint。把值写成 `default` 不是合法家族，会失败。 |
| `superpoint` | SuperPoint + SuperPoint LightGlue |
| `aliked` | ALIKED + ALIKED LightGlue |
| `superpoint_flexglue` | SuperPoint 关键点，FlexGlue 稠密特征，FlexGlue 匹配器 |
| `aliked_flexglue` | ALIKED 关键点，FlexGlue 稠密特征，FlexGlue 匹配器 |

其他字符串打印 `Unsupported feature family`。`GlobalRetriever::Create` 接受 `""`、`"default"`、`"cosplace"` 和 `"eigenplaces"`。`""` 和 `"default"` 仍然选 CosPlace。其他字符串打印 `Unsupported global retriever type`。

图像是 `cv::Mat`。1 通道当灰度，3 通道当 BGR，4 通道当 BGRA。别的通道数会失败。SuperPoint 会转成灰度。ALIKED、CosPlace、EigenPlaces、FlexGlue 转成 RGB。流水线把 SuperPoint 和 ALIKED 缩放到 `image_width` × `image_height`，再把关键点乘回输入图的坐标。CosPlace 和 EigenPlaces 各自缩放到自己的 `input_size` × `input_size`。FlexGlue 描述子缩放到它自己的 `input_size`。

运行时代码会读这些项：

- `feature_type`：`Create("")` 选哪一家。示例是 `superpoint`。
- `superpoint.image_width`、`image_height`：示例是 `512` 和 `288`。流水线先缩到这个尺寸，engine 也按这个尺寸跑。默认 ONNX 导出是固定的 512×288，改 yaml 就要一起重新导出。对不上时 `setInputShape` 失败。
- `superpoint.max_keypoints`、`keypoint_threshold`（`0.01`）、`remove_borders`（`4`）：网络跑完后在 C++ 里用。
- `superpoint_lightglue.filter_threshold`（`0.1`）。`num_keypoints` 可以不写；不写就抄 `superpoint.max_keypoints`。默认 LightGlue 导出是 1024 个点、固定形状。这个数要和导出的 engine 一致。
- `aliked.image_width`、`image_height`：和 SuperPoint 一样，默认 512×288。`descriptor_dim` 必须是 `128`，否则 `Create` 返回空指针。运行时的点数用 engine 的输出形状。默认导出文件名是 `top1024`；只改 yaml 里的 `top_k` 不会改这个形状。
- `aliked_lightglue.descriptor_dim` 必须是 `128`。示例里 `filter_threshold` 是 `0.1`。
- `cosplace.input_size` 是 `320`，`descriptor_dim` 是 `512`。engine 的描述子长度必须等于 `descriptor_dim`。
- `eigenplaces.input_size` 是 `512`，`descriptor_dim` 是 `512`。张量名和 CosPlace 一样（`images`、`descriptors`），ONNX 和 engine 文件不同。engine 的描述子长度必须等于 `descriptor_dim`。512 这条边对齐 EigenPlaces 的训练 crop。改了就要重新导出；GeM 的池化核在导出时就固定了。
- `flexglue`：`input_size` 512，`feature_dim` 256，`feature_width` 和 `feature_height` 64，`max_keypoints` 1024（可以不写，结构体默认就是 1024），`filter_threshold` 0.1，`max_distance_px` 3，`min_covisibility` 0。`./build_trt.sh` 按这个特征图尺寸、最多 1024 个点来编匹配器。点数超过 `max_keypoints` 会打印 `Too many keypoints for the configured profile`。FlexGlue 没有 `onnx_file`，也没有 `dla_core`。
- 其他模型的 `dla_core`：大于等于 0 时，库从 ONNX 编 engine 会打开 DLA。示例是 `-1`，不打开。机器上没有 DLA 核心时，填一个核心编号会先打印错误，然后走到 `assert`。

张量名列表必须和 engine 一致。示例里的名字和 `convert2onnx/` 的导出器一致。名字不对会报 `Unexpected network IO count` 或 `Unexpected tensor layout`。

`LocalFeatures` 里有输入图坐标下的 `keypoints`，N×D 的 `CV_32F` `descriptors`（SuperPoint 的 D 是 256，ALIKED 是 128），`image_size`，以及 `family`（`"superpoint"` 或 `"aliked"`）。它只能移动，不能拷贝。`Match` 会拒绝来自另一条流水线的特征。

## 跑示例

用 `-DBUILD_EXAMPLES=ON` 从仓库根目录编，或者在 `example/` 里编。下面的命令按根目录那次构建来写。如果二进制在 `example/build`，把路径里的 `build/example` 换成 `example/build`。

`inference_image` 匹配两张图。用法：

```text
inference_image first_image_path second_image_path [output_image_path] [iterations] [superpoint|aliked|superpoint_flexglue|aliked_flexglue] [width] [height]
```

默认：输出写到当前目录的 `match_image.png`，迭代 100 次，家族 `superpoint`，缩放到 512×288。打印的平均时间不算第一次。图上的标题固定是 `SuperPoint LightGlue TensorRT`；你传入的家族名打在标准输出上。

```bash
mkdir -p example/output
./build/example/inference_image \
  example/image/image0.png example/image/image1.png \
  example/output/superpoint.png 10 superpoint
./build/example/inference_image \
  example/image/image0.png example/image/image1.png \
  example/output/aliked_flexglue.png 10 aliked_flexglue
```

`inference_sequence` 把每一帧和下一帧匹配，写出 `<i-1>_<i>.png`。目录里至少要有两张图。程序的用法字符串把目录叫成绝对路径；相对路径可以用。

```text
inference_sequence image_folder output_folder [superpoint|aliked|superpoint_flexglue|aliked_flexglue] [width] [height]
```

```bash
./build/example/inference_sequence \
  example/image/freiburg_sequence \
  example/output/sequence_superpoint \
  superpoint
```

`inference_cosplace` 用第 0 帧和后面每一帧比。它保留 `.jpg`、`.jpeg`、`.png`、`.bmp`。用法：

```text
inference_cosplace <image_sequence_dir> [--threshold <float>] [--output <csv>] [--max-images <int>]
```

默认：阈值 `0.80`，CSV 写到当前目录的 `cosplace_loop_similarity.csv`，`--max-images 0` 表示不限制张数。阈值必须在 [-1, 1]。CSV 列是 `frame_index,image,similarity,is_loop`。

```bash
./build/example/inference_cosplace \
  example/image/freiburg_sequence \
  --threshold 0.80 \
  --output example/output/cosplace_loop_similarity.csv
```

`inference_eigenplaces` 做同样的比较，调用的是 `GlobalRetriever::Create("eigenplaces")`。用法：

```text
inference_eigenplaces <image_sequence_dir> [--threshold <float>] [--output <csv>] [--max-images <int>]
```

默认：阈值 `0.80`，CSV 写到当前目录的 `eigenplaces_loop_similarity.csv`，`--max-images 0` 表示不限制张数。这个阈值只决定 CSV 里怎么标回环，和 `inference_cosplace` 的开关默认值相同，不是在 EigenPlaces 权重上测出来的分数。

```bash
./build/example/inference_eigenplaces \
  example/image/freiburg_sequence \
  --threshold 0.80 \
  --output example/output/eigenplaces_loop_similarity.csv
```

输出路径如果是相对的，就相对当前目录。配置和 `weights/` 不看当前目录，按上面的搜索顺序找，所以这些程序不必从仓库根目录启动。

`example/utils.h` 只给示例用（列文件、`cv::drawMatches`），不是库的一部分。更多 CMake 片段在 `example/LINKING.md`。

## 接到自己的程序

`cmake --install` 之后：

```cmake
find_package(viewfeat REQUIRED)
target_link_libraries(my_app PRIVATE viewfeat::viewfeat)
```

不安装、放在同一次构建里：

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

`image0` 和 `image1` 是 `cv::Mat`。`descriptor.size()` 等于 `retriever->DescriptorDim()`，示例配置下 CosPlace 和 EigenPlaces 都是 512。运行时进程还要能找到 `libviewfeat.so`、`libtensorrtbuffer.so`，以及 TensorRT 和 CUDA 的库。示例程序会把前两个拷到可执行文件旁边。安装后的库，RPATH 包含 `$ORIGIN` 和编译时找到的 TensorRT 库目录。

`FeaturePipeline` 上还有这些方法：带关键点和描述子矩阵的 `Match`（只给 LightGlue 用；FlexGlue 需要稠密特征图，这个重载返回 false）、`MatchImages`，以及 `MatchWithImages`（FlexGlue，要两张图和关键点）。`Family()` 返回流水线或检索器的名字。`DescriptorDim()` 对 SuperPoint 是 256，对 ALIKED 是 128。

## 测试

完整 ctest 在镜像 `visual-localization-trt:blackwell-20261114` 里跑，并且要看见 GPU。下面这条命令是在有 RTX 5070 Ti 的主机上用的。必须加 `--gpus all`。不加的话容器里没有显卡：`engine_io` 和 `config_load` 仍可能通过，五个图像测试以 77 退出，ctest 把这记成跳过。

```bash
docker run --rm --gpus all -v "$PWD":/src -w /src visual-localization-trt:blackwell-20261114 \
  bash -lc 'nvidia-smi -L && cmake -S /src -B /tmp/viewfeat-build -G Ninja -DBUILD_TESTING=ON && cmake --build /tmp/viewfeat-build && ctest --test-dir /tmp/viewfeat-build --output-on-failure -j1'
```

先跑 `nvidia-smi -L`，输出里应该有这块 GPU。图像测试还要仓库里的 `weights/`（挂到 `/src/weights`），以及这些已经在树里的文件：

- `example/image/image0.png`
- `example/image/image1.png`
- `example/image/freiburg_sequence/1341847980.722988.png`
- `example/image/freiburg_sequence/1341847981.726650.png`

`weights/` 或其中某个文件不在，图像测试以 77 退出。`weights/` 在，但那个模型的 engine 或 ONNX 加载失败，测试就是失败。ctest 不需要 `-DBUILD_EXAMPLES=ON`。名字以 `example_` 开头的测试来自 `test/test_images.cpp`，不是示例程序。

| ctest 名 | 命令 | GPU | 超时 |
| --- | --- | --- | --- |
| `engine_io` | `test_engine_io` | 不需要 | 60 秒 |
| `engine_io_garbage` | `test_engine_io --garbage` | 走反序列化；没有 NVIDIA 设备且 TensorRT 抛异常或被信号杀掉时，退出码 77 | 90 秒 |
| `config_load` | `test_config` | 不需要 | 60 秒 |
| `example_superpoint` | `test_images default`，加上配置、`weights/` 和那四张图 | 需要，否则退出 77 | 180 秒 |
| `example_aliked` | `test_images aliked`，路径相同 | 需要，否则退出 77 | 180 秒 |
| `example_cosplace` | `test_images cosplace`，路径相同 | 需要，否则退出 77 | 180 秒 |
| `example_eigenplaces` | `test_images eigenplaces`，路径相同 | 需要，否则退出 77 | 300 秒 |
| `example_flexglue` | `test_images superpoint_flexglue`，路径相同 | 需要，否则退出 77 | 300 秒 |

`example_flexglue` 不跑 `aliked_flexglue`。`/dev/nvidia0`、`/dev/nvidiactl`、`/dev/dxg` 里有一个存在，就算有设备。上面的命令带 `-j1`。

不进 Docker、本机依赖已经装好、GPU 可见时：

```bash
nvidia-smi -L
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure -j1
```

## 目录

```text
include/viewfeat/          公开头文件
src/                       流水线、检索器、局部特征、模型、运行时
config/config.yaml         尺寸、阈值、张量名、权重文件名
example/                   四个程序、example/image、链接说明
example/image/             image0.png、image1.png、freiburg_sequence/
test/                      engine、配置和图像测试
convert2onnx/              python -m onnx_export
export_onnx.sh             把 SuperPoint、ALIKED、LightGlue、CosPlace、EigenPlaces 导出到 weights/
build_trt.sh               对 weights/ 里的 ONNX 跑 trtexec
docker/Dockerfile          基于 TensorRT 25.03 的镜像
3rdparty/tensorrtbuffer/   TensorRT 示例辅助代码，编成 libtensorrtbuffer.so
cmake/                     CMake 包模板
package.xml                ROS 2 包名 viewfeat
docs/                      内部审查记录，不是使用说明
```
