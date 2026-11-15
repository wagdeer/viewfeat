#include "models/flexglue.h"

#include "models/outlier_rejection.h"
#include "runtime/engine_io.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>

#include <opencv2/imgproc.hpp>

using samplesCommon::BufferManager;
using samplesCommon::HostMemoryType;

namespace viewfeat {

namespace {

bool BindAllTensors(const std::shared_ptr<nvinfer1::ICudaEngine> &engine, nvinfer1::IExecutionContext &context, BufferManager &buffers) {
  for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
    const char *name = engine->getIOTensorName(i);
    if (!context.setTensorAddress(name, buffers.getDeviceBuffer(name))) {
      return false;
    }
  }
  return true;
}

}  // namespace

FlexGlueDescriptor::FlexGlueDescriptor(FlexGlueConfig config) : config_(std::move(config)), stream_(nullptr) {
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    stream_ = nullptr;
    std::cerr << "[FlexGlueDescriptor] Failed to create CUDA stream." << std::endl;
  }
}

FlexGlueDescriptor::~FlexGlueDescriptor() {
  if (stream_ != nullptr) {
    cudaStreamSynchronize(stream_);
  }
  context_.reset();
  buffers_.reset();
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

bool FlexGlueDescriptor::Build() {
  if (DeserializeEngine()) {
    return true;
  }
  return false;
}

bool FlexGlueDescriptor::DeserializeEngine() {
  engine_ = LoadSerializedEngine(config_.descriptor_engine_file, "FlexGlueDescriptor");
  if (!engine_) {
    return false;
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    std::cerr << "[FlexGlueDescriptor] Failed to create execution context for " << config_.descriptor_engine_file << "." << std::endl;
    engine_.reset();
    return false;
  }
  if (config_.descriptor_input_tensor_names.size() != 1 || config_.descriptor_output_tensor_names.size() != 1) {
    std::cerr << "[FlexGlueDescriptor] Invalid tensor-name configuration for " << config_.descriptor_engine_file << "." << std::endl;
    engine_.reset();
    context_.reset();
    return false;
  }
  return true;
}

bool FlexGlueDescriptor::Infer(const cv::Mat &image, std::vector<float> &features) {
  features.clear();
  if (image.empty()) {
    return false;
  }
  if (!context_) {
    std::cerr << "[FlexGlueDescriptor] Execution context is missing for " << config_.descriptor_engine_file << "." << std::endl;
    return false;
  }
  if (stream_ == nullptr) {
    std::cerr << "[FlexGlueDescriptor] CUDA stream is missing." << std::endl;
    return false;
  }

  const char *input_name = config_.descriptor_input_tensor_names[0].c_str();
  const nvinfer1::Dims4 input_shape(1, 3, config_.input_size, config_.input_size);
  if (!context_->setInputShape(input_name, input_shape)) {
    return false;
  }

  if (!EnsureBuffers() || !ProcessInput(*buffers_, image)) {
    return false;
  }
  buffers_->copyInputToDeviceAsync(stream_);
  if (!context_->enqueueV3(stream_)) {
    cudaStreamSynchronize(stream_);
    std::cerr << "[FlexGlueDescriptor] enqueueV3() failed." << std::endl;
    return false;
  }
  buffers_->copyOutputToHostAsync(stream_);
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    std::cerr << "[FlexGlueDescriptor] cudaStreamSynchronize() failed." << std::endl;
    return false;
  }

  const size_t expected =
      static_cast<size_t>(config_.feature_dim) * config_.feature_height * config_.feature_width;
  const auto *output =
      static_cast<const float *>(buffers_->getHostBuffer(config_.descriptor_output_tensor_names[0]));
  if (output == nullptr) {
    return false;
  }
  features.assign(output, output + expected);
  return true;
}

bool FlexGlueDescriptor::EnsureBuffers() {
  if (buffers_) {
    return true;
  }
  buffers_ = std::make_unique<BufferManager>(engine_, 0, context_.get(), HostMemoryType::kPinned);
  if (!buffers_ || !BindAllTensors(engine_, *context_, *buffers_)) {
    buffers_.reset();
    return false;
  }
  return true;
}

bool FlexGlueDescriptor::ProcessInput(const BufferManager &buffers, const cv::Mat &image) {
  cv::Mat rgb;
  if (image.channels() == 1) {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
  } else {
    return false;
  }

  cv::resize(rgb, rgb, cv::Size(config_.input_size, config_.input_size), 0.0, 0.0, cv::INTER_LINEAR);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
  auto *input = static_cast<float *>(buffers.getHostBuffer(config_.descriptor_input_tensor_names[0]));
  const int channel_size = config_.input_size * config_.input_size;
  for (int y = 0; y < config_.input_size; ++y) {
    for (int x = 0; x < config_.input_size; ++x) {
      const cv::Vec3f pixel = rgb.at<cv::Vec3f>(y, x);
      const int offset = y * config_.input_size + x;
      for (int c = 0; c < 3; ++c) {
        input[c * channel_size + offset] = pixel[c];
      }
    }
  }
  return true;
}

FlexGlueMatcher::FlexGlueMatcher(FlexGlueConfig config) : config_(std::move(config)), stream_(nullptr) {
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    stream_ = nullptr;
    std::cerr << "[FlexGlueMatcher] Failed to create CUDA stream." << std::endl;
  }
}

FlexGlueMatcher::~FlexGlueMatcher() {
  if (stream_ != nullptr) {
    cudaStreamSynchronize(stream_);
  }
  context_.reset();
  buffers_.reset();
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

bool FlexGlueMatcher::Build() {
  if (DeserializeEngine()) {
    return true;
  }
  return false;
}

bool FlexGlueMatcher::DeserializeEngine() {
  engine_ = LoadSerializedEngine(config_.matcher_engine_file, "FlexGlueMatcher");
  if (!engine_) {
    return false;
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    std::cerr << "[FlexGlueMatcher] Failed to create execution context for " << config_.matcher_engine_file << "." << std::endl;
    engine_.reset();
    return false;
  }
  if (config_.matcher_input_tensor_names.size() != 9 || config_.matcher_output_tensor_names.size() != 3) {
    std::cerr << "[FlexGlueMatcher] Invalid tensor-name configuration for " << config_.matcher_engine_file << "." << std::endl;
    engine_.reset();
    context_.reset();
    return false;
  }
  return true;
}

bool FlexGlueMatcher::Infer(const std::vector<float> &features0, const std::vector<float> &features1,
                            const std::vector<cv::Point2f> &keypoints0, const std::vector<cv::Point2f> &keypoints1,
                            int image0_width, int image0_height, int image1_width, int image1_height,
                            std::vector<cv::DMatch> &matches, bool outlier_rejection, float *covisibility_score) {
  matches.clear();
  if (covisibility_score != nullptr) {
    *covisibility_score = 0.0F;
  }
  if (!context_) {
    std::cerr << "[FlexGlueMatcher] Execution context is missing for " << config_.matcher_engine_file << "." << std::endl;
    return false;
  }
  if (stream_ == nullptr) {
    std::cerr << "[FlexGlueMatcher] CUDA stream is missing." << std::endl;
    return false;
  }
  if (keypoints0.size() < 2 || keypoints1.size() < 2) {
    return true;
  }
  if (keypoints0.size() > static_cast<size_t>(config_.max_keypoints) ||
      keypoints1.size() > static_cast<size_t>(config_.max_keypoints)) {
    std::cerr << "[FlexGlueMatcher] Too many keypoints for the configured profile." << std::endl;
    return false;
  }

  const auto &names = config_.matcher_input_tensor_names;
  if (!context_->setInputShape(names[2].c_str(), nvinfer1::Dims3(1, static_cast<int32_t>(keypoints0.size()), 2)) ||
      !context_->setInputShape(names[3].c_str(), nvinfer1::Dims3(1, static_cast<int32_t>(keypoints1.size()), 2))) {
    return false;
  }

  if (!EnsureBuffers(static_cast<int>(keypoints0.size()), static_cast<int>(keypoints1.size()))) {
    return false;
  }

  const size_t feature_elements =
      static_cast<size_t>(config_.feature_dim) * config_.feature_height * config_.feature_width;
  if (features0.size() != feature_elements || features1.size() != feature_elements) {
    return false;
  }
  auto *kpts0 = static_cast<float *>(buffers_->getHostBuffer(names[2]));
  auto *kpts1 = static_cast<float *>(buffers_->getHostBuffer(names[3]));
  for (size_t i = 0; i < keypoints0.size(); ++i) {
    kpts0[2 * i] = keypoints0[i].x + 0.5F;
    kpts0[2 * i + 1] = keypoints0[i].y + 0.5F;
  }
  for (size_t i = 0; i < keypoints1.size(); ++i) {
    kpts1[2 * i] = keypoints1[i].x + 0.5F;
    kpts1[2 * i + 1] = keypoints1[i].y + 0.5F;
  }

  auto *size0 = static_cast<float *>(buffers_->getHostBuffer(names[4]));
  auto *size1 = static_cast<float *>(buffers_->getHostBuffer(names[5]));
  size0[0] = static_cast<float>(image0_width);
  size0[1] = static_cast<float>(image0_height);
  size1[0] = static_cast<float>(image1_width);
  size1[1] = static_cast<float>(image1_height);
  *static_cast<float *>(buffers_->getHostBuffer(names[6])) = static_cast<float>(config_.filter_threshold);
  *static_cast<float *>(buffers_->getHostBuffer(names[7])) = static_cast<float>(config_.max_distance_px);
  *static_cast<float *>(buffers_->getHostBuffer(names[8])) = static_cast<float>(config_.min_covisibility);

  const size_t feature_bytes = feature_elements * sizeof(float);
  if (cudaMemcpyAsync(buffers_->getDeviceBuffer(names[0]), features0.data(), feature_bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
      cudaMemcpyAsync(buffers_->getDeviceBuffer(names[1]), features1.data(), feature_bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
    std::cerr << "[FlexGlueMatcher] Failed to copy feature maps." << std::endl;
    return false;
  }
  for (size_t index = 2; index < names.size(); ++index) {
    const size_t bytes = buffers_->size(names[index]);
    void *device = buffers_->getDeviceBuffer(names[index]);
    void *host = buffers_->getHostBuffer(names[index]);
    if (device == nullptr || host == nullptr || bytes == 0 || bytes == BufferManager::kINVALID_SIZE_VALUE) {
      return false;
    }
    if (cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
      std::cerr << "[FlexGlueMatcher] Failed to copy " << names[index] << "." << std::endl;
      return false;
    }
  }
  if (!context_->enqueueV3(stream_)) {
    cudaStreamSynchronize(stream_);
    std::cerr << "[FlexGlueMatcher] enqueueV3() failed." << std::endl;
    return false;
  }
  buffers_->copyOutputToHostAsync(stream_);
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    std::cerr << "[FlexGlueMatcher] cudaStreamSynchronize() failed." << std::endl;
    return false;
  }

  const auto *matches0 =
      static_cast<const int64_t *>(buffers_->getHostBuffer(config_.matcher_output_tensor_names[0]));
  const auto *scores =
      static_cast<const float *>(buffers_->getHostBuffer(config_.matcher_output_tensor_names[1]));
  const auto *covisibility =
      static_cast<const float *>(buffers_->getHostBuffer(config_.matcher_output_tensor_names[2]));
  if (matches0 == nullptr || scores == nullptr || covisibility == nullptr) {
    return false;
  }

  std::vector<cv::Point2f> points0;
  std::vector<cv::Point2f> points1;
  for (size_t i = 0; i < keypoints0.size(); ++i) {
    const int64_t j = matches0[i];
    if (j < 0 || j >= static_cast<int64_t>(keypoints1.size())) {
      continue;
    }
    matches.emplace_back(static_cast<int>(i), static_cast<int>(j), scores[i]);
    points0.push_back(keypoints0[i]);
    points1.push_back(keypoints1[static_cast<size_t>(j)]);
  }
  if (covisibility_score != nullptr) {
    *covisibility_score = covisibility[0];
  }

  if (outlier_rejection) {
    RejectFundamentalOutliers(points0, points1, matches);
  }
  return true;
}

bool FlexGlueMatcher::EnsureBuffers(int keypoints0, int keypoints1) {
  if (buffers_ && buffer_keypoints0_ == keypoints0 && buffer_keypoints1_ == keypoints1) {
    return true;
  }
  buffers_ = std::make_unique<BufferManager>(engine_, 0, context_.get(), HostMemoryType::kPinned);
  if (!buffers_ || !BindAllTensors(engine_, *context_, *buffers_)) {
    buffers_.reset();
    buffer_keypoints0_ = 0;
    buffer_keypoints1_ = 0;
    return false;
  }
  buffer_keypoints0_ = keypoints0;
  buffer_keypoints1_ = keypoints1;
  return true;
}

}  // namespace viewfeat
