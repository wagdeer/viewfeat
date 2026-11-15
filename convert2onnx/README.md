# ONNX Export (`convert2onnx`)

在仓库里进入本目录后直接运行：

```bash
cd /workspace/viewfeat/convert2onnx
python3 -m onnx_export export all   # SuperPoint + ALIKED + LightGlue + CosPlace + EigenPlaces
python3 -m onnx_export export eigenplaces
python3 -m onnx_export compare --inline ../weights/foo.onnx
python3 -m onnx_export trt --weights-dir ../weights
python3 -m onnx_export trt --weights-dir ../weights --models cosplace eigenplaces
```

## 目录结构

```text
convert2onnx/
├── onnx_export/          # 导出管线、CLI、对比工具
└── models/               # vendored PyTorch 模型（仅导出用）
```

默认会做 `onnxsim` 简化，需要环境里已有 `onnxsim` 和 `onnxruntime`（与重构前相同）。
