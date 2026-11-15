#!/usr/bin/env bash
# Build TensorRT engines from ONNX files in weights/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}/convert2onnx"
exec python3 -m onnx_export trt --weights-dir "${ROOT}/weights" "$@"
