#include "runtime/engine_io.h"

#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "buffers.h"
#include "logger.h"

using samplesCommon::SampleUniquePtr;

namespace viewfeat {

namespace {

const char *EngineLabel(const char *label) { return label != nullptr ? label : "Engine"; }

}  // namespace

std::shared_ptr<nvinfer1::ICudaEngine> LoadSerializedEngine(const std::string &path, const char *label) {
  const char *name = EngineLabel(label);
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[" << name << "] Engine not found: " << path << "." << std::endl;
    return nullptr;
  }
  file.seekg(0, std::ifstream::end);
  const std::streamoff end = file.tellg();
  if (end <= 0) {
    std::cerr << "[" << name << "] Failed to read engine: " << path << "." << std::endl;
    return nullptr;
  }
  file.seekg(0, std::ifstream::beg);
  const size_t size = static_cast<size_t>(end);
  std::vector<char> stream(size);
  file.read(stream.data(), static_cast<std::streamsize>(size));
  if (!file) {
    std::cerr << "[" << name << "] Failed to read engine: " << path << "." << std::endl;
    return nullptr;
  }

  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
  if (!runtime) {
    return nullptr;
  }
  auto *engine = runtime->deserializeCudaEngine(stream.data(), size);
  if (engine == nullptr) {
    std::cerr << "[" << name << "] Failed to deserialize engine: " << path
              << " (TensorRT version or GPU architecture mismatch)." << std::endl;
    return nullptr;
  }
  return std::shared_ptr<nvinfer1::ICudaEngine>(engine);
}

std::shared_ptr<nvinfer1::ICudaEngine> BuildEngineFromOnnx(const std::string &onnx_path, int dla_core, const char *label,
                                                            int optimization_level) {
  const char *name = EngineLabel(label);
  auto fail = [&]() {
    std::cerr << "[" << name << "] Failed to build engine from " << onnx_path << "." << std::endl;
    return std::shared_ptr<nvinfer1::ICudaEngine>();
  };

  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  if (!builder) {
    return fail();
  }
  auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
  if (!network) {
    return fail();
  }
  auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return fail();
  }
  auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
  if (!parser) {
    return fail();
  }
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()))) {
    std::cerr << "[" << name << "] Failed to parse ONNX: " << onnx_path << "." << std::endl;
    return nullptr;
  }
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  if (optimization_level >= 0) {
    config->setBuilderOptimizationLevel(optimization_level);
  }
  samplesCommon::enableDLA(builder.get(), config.get(), dla_core);

  auto profile_stream = samplesCommon::makeCudaStream();
  if (!profile_stream) {
    return fail();
  }
  config->setProfileStream(*profile_stream);
  SampleUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    return fail();
  }
  SampleUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
  if (!runtime) {
    return fail();
  }
  std::shared_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine) {
    return fail();
  }
  return engine;
}

bool SaveSerializedEngine(const nvinfer1::ICudaEngine &engine, const std::string &path) {
  std::unique_ptr<nvinfer1::IHostMemory> data(engine.serialize());
  if (!data) {
    std::cerr << "[Engine] Failed to serialize engine: " << path << "." << std::endl;
    return false;
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "[Engine] Failed to write engine: " << path << "." << std::endl;
    return false;
  }
  file.write(reinterpret_cast<const char *>(data->data()), static_cast<std::streamsize>(data->size()));
  if (!file) {
    std::cerr << "[Engine] Failed to write engine: " << path << "." << std::endl;
    return false;
  }
  return true;
}

}  // namespace viewfeat
