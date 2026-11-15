#!/usr/bin/env bash
# Backward-compatible wrapper. Prefer: python -m onnx_export trt --weights-dir ../weights
set -euo pipefail
cd "$(dirname "$0")"
WEIGHTS_DIR="${WEIGHTS_DIR:-../weights}"
exec python -m onnx_export trt --weights-dir "${WEIGHTS_DIR}" "$@"
