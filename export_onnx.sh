#!/usr/bin/env bash
# Export all default ONNX models (SuperPoint + ALIKED + LightGlue + CosPlace + EigenPlaces).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}/convert2onnx"
exec python3 -m onnx_export export all "$@"
