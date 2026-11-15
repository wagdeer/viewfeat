#ifndef VIEWFEAT_RUNTIME_ENGINE_IO_H_
#define VIEWFEAT_RUNTIME_ENGINE_IO_H_

#include <NvInfer.h>

#include <memory>
#include <string>

namespace viewfeat {

std::shared_ptr<nvinfer1::ICudaEngine> LoadSerializedEngine(const std::string &path, const char *label);
bool SaveSerializedEngine(const nvinfer1::ICudaEngine &engine, const std::string &path);

// Parses ONNX, builds an FP16 engine, and deserializes it. Returns nullptr on failure.
// optimization_level < 0 keeps the TensorRT default.
std::shared_ptr<nvinfer1::ICudaEngine> BuildEngineFromOnnx(const std::string &onnx_path, int dla_core, const char *label,
                                                            int optimization_level = -1);

}  // namespace viewfeat

#endif  // VIEWFEAT_RUNTIME_ENGINE_IO_H_
