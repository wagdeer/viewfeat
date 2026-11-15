FROM nvcr.io/nvidia/tensorrt:25.03-py3

# TensorRT 25.03 provides Ubuntu 24.04 + Python 3.12 + CUDA 12.8.1 +
# TensorRT 10.9. CUDA 12.8 is required for Blackwell GPUs such as the
# RTX 5070 Ti (compute capability 12.0).
ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_MIRROR=mirrors.aliyun.com
ARG UBUNTU_SECURITY_MIRROR=mirrors.aliyun.com
ARG PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple/
ARG PYTORCH_WHEEL_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cu128/
ARG TORCH_VERSION=2.7.0+cu128
ARG TORCHVISION_VERSION=0.22.0+cu128

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

SHELL ["/bin/bash", "-c"]

# Use a nearby mirror when the base image exposes Ubuntu sources in either
# classic sources.list or newer deb822 ubuntu.sources format.
RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
      sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|https://${UBUNTU_MIRROR}/ubuntu|g" \
        -e "s|http://security.ubuntu.com/ubuntu|https://${UBUNTU_SECURITY_MIRROR}/ubuntu|g" \
        /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
      sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|https://${UBUNTU_MIRROR}/ubuntu|g" \
        -e "s|http://security.ubuntu.com/ubuntu|https://${UBUNTU_SECURITY_MIRROR}/ubuntu|g" \
        /etc/apt/sources.list.d/ubuntu.sources; \
    fi

# Build toolchain and runtime dependencies for this project only.
RUN set -eux; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get -o Acquire::Retries=5 install --no-install-recommends -y \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libeigen3-dev \
      libopencv-dev \
      libyaml-cpp-dev \
      ninja-build \
      pkg-config \
      python3-dev \
      python3-pip \
      unzip \
      vim \
      wget; \
    rm -rf /var/lib/apt/lists/*

# Model-export dependencies for pt -> ONNX -> TensorRT engine workflows.
# Torch 2.7 is the first release line with official cu128 wheels required by
# Blackwell consumer GPUs.
RUN set -eux; \
    python3 -m pip install --no-cache-dir --break-system-packages \
      --index-url "${PYPI_INDEX}" \
      onnx \
      onnxruntime \
      onnxsim; \
    python3 -m pip install --no-cache-dir --break-system-packages \
      --index-url "${PYPI_INDEX}" \
      --ignore-installed "sympy>=1.13.1"; \
    python3 -m pip install --no-cache-dir --break-system-packages \
      --index-url "${PYPI_INDEX}" \
      --find-links "${PYTORCH_WHEEL_INDEX}" \
      torch==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION}

# Validate the Python model-export stack without requiring GPU access at build
# time. GPU/CUDA capability is checked at container runtime.
RUN python3 -c "import onnx, onnxruntime, tensorrt, torch, torchvision; print('onnx', onnx.__version__); print('onnxruntime', onnxruntime.__version__); print('TensorRT', tensorrt.__version__); print('torch', torch.__version__, 'CUDA', torch.version.cuda); print('torchvision', torchvision.__version__)"

WORKDIR /workspace
