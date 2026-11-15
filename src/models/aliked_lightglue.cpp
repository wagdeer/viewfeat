#include "models/aliked_lightglue.h"

#include "models/lightglue_postprocess.h"
#include "models/outlier_rejection.h"
#include "runtime/engine_io.h"

#include <cmath>
#include <iostream>
#include <utility>

namespace viewfeat {

using sample::gLogger;
using samplesCommon::BufferManager;
using samplesCommon::SampleUniquePtr;

constexpr int AlikedLightGlue::kDescriptorDim;
constexpr int AlikedLightGlue::kFeatureRows;

AlikedLightGlue::AlikedLightGlue(AlikedLightGlueConfig aliked_lightglue_config)
    : aliked_lightglue_config_(std::move(aliked_lightglue_config)), engine_(nullptr), stream_(nullptr) {
  sample::setReportableSeverity(sample::Logger::Severity::kINTERNAL_ERROR);
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    stream_ = nullptr;
    std::cerr << "[AlikedLightGlue] Failed to create CUDA stream." << std::endl;
  }
}

AlikedLightGlue::~AlikedLightGlue() {
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

bool AlikedLightGlue::Build() {
  if (DeserializeEngine()) {
    buffers_.reset();
    buffer_keypoints0_ = 0;
    buffer_keypoints1_ = 0;
    return true;
  }

  engine_ = BuildEngineFromOnnx(aliked_lightglue_config_.onnx_file, aliked_lightglue_config_.dla_core, "AlikedLightGlue");
  if (!engine_) {
    std::cerr << "[AlikedLightGlue] Failed to build engine " << aliked_lightglue_config_.engine_file << "." << std::endl;
    return false;
  }

  SaveEngine();
  if (engine_->getNbIOTensors() != 5 || aliked_lightglue_config_.input_tensor_names.size() != 4 ||
      aliked_lightglue_config_.output_tensor_names.size() != 1) {
    std::cerr << "[AlikedLightGlue] Expected 4 network inputs for " << aliked_lightglue_config_.engine_file << "." << std::endl;
    return false;
  }
  keypoints_0_dims_ = engine_->getTensorShape(aliked_lightglue_config_.input_tensor_names[0].c_str());
  keypoints_1_dims_ = engine_->getTensorShape(aliked_lightglue_config_.input_tensor_names[1].c_str());
  buffers_.reset();
  buffer_keypoints0_ = 0;
  buffer_keypoints1_ = 0;
  return true;
}

bool AlikedLightGlue::EnsureBuffers(int keypoints0, int keypoints1) {
  if (buffers_ && buffer_keypoints0_ == keypoints0 && buffer_keypoints1_ == keypoints1) {
    return true;
  }

  buffers_ = std::make_unique<BufferManager>(engine_, 0, context_.get(), samplesCommon::HostMemoryType::kPinned);
  if (!buffers_) {
    return false;
  }

  buffer_keypoints0_ = keypoints0;
  buffer_keypoints1_ = keypoints1;
  for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; ++i) {
    auto const tensor_name = engine_->getIOTensorName(i);
    if (!context_->setTensorAddress(tensor_name, buffers_->getDeviceBuffer(tensor_name))) {
      buffers_.reset();
      buffer_keypoints0_ = 0;
      buffer_keypoints1_ = 0;
      return false;
    }
  }
  return true;
}

bool AlikedLightGlue::Infer(const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features0, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features1,
                            Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score, int image0_width, int image0_height,
                            int image1_width, int image1_height) {
  if (!context_) {
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      std::cerr << "[AlikedLightGlue] Failed to create execution context for " << aliked_lightglue_config_.engine_file << "." << std::endl;
      return false;
    }
  }
  if (stream_ == nullptr) {
    std::cerr << "[AlikedLightGlue] CUDA stream is missing." << std::endl;
    return false;
  }

  if (engine_->getNbIOTensors() != 5 || aliked_lightglue_config_.input_tensor_names.size() != 4 ||
      aliked_lightglue_config_.output_tensor_names.size() != 1) {
    std::cerr << "[AlikedLightGlue] Unexpected tensor layout for " << aliked_lightglue_config_.engine_file << "." << std::endl;
    return false;
  }

  auto const keypoints_0_name = aliked_lightglue_config_.input_tensor_names[0].c_str();
  auto const keypoints_1_name = aliked_lightglue_config_.input_tensor_names[1].c_str();
  auto const descriptors_0_name = aliked_lightglue_config_.input_tensor_names[2].c_str();
  auto const descriptors_1_name = aliked_lightglue_config_.input_tensor_names[3].c_str();

  if (!context_->setInputShape(keypoints_0_name, nvinfer1::Dims3(1, features0.cols(), 2))) {
    return false;
  }
  if (!context_->setInputShape(keypoints_1_name, nvinfer1::Dims3(1, features1.cols(), 2))) {
    return false;
  }
  if (!context_->setInputShape(descriptors_0_name, nvinfer1::Dims3(1, features0.cols(), kDescriptorDim))) {
    return false;
  }
  if (!context_->setInputShape(descriptors_1_name, nvinfer1::Dims3(1, features1.cols(), kDescriptorDim))) {
    return false;
  }

  if (!EnsureBuffers(features0.cols(), features1.cols())) {
    return false;
  }

  keypoints_0_dims_ = context_->getTensorShape(keypoints_0_name);
  keypoints_1_dims_ = context_->getTensorShape(keypoints_1_name);

  if (!ProcessInput(*buffers_, features0, features1, image0_width, image0_height, image1_width, image1_height)) {
    return false;
  }

  buffers_->copyInputToDeviceAsync(stream_);

  bool status = context_->enqueueV3(stream_);
  if (!status) {
    cudaStreamSynchronize(stream_);
    return false;
  }

  buffers_->copyOutputToHostAsync(stream_);
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    std::cerr << "[AlikedLightGlue] cudaStreamSynchronize() failed." << std::endl;
    return false;
  }
  if (!ProcessOutput(*buffers_, matches_index, matches_score)) {
    return false;
  }
  return true;
}

bool AlikedLightGlue::ProcessInput(const BufferManager &buffers, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features0, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features1,
                                    int image0_width, int image0_height, int image1_width, int image1_height) {
  auto *keypoints_0_buffer = static_cast<float *>(buffers.getHostBuffer(aliked_lightglue_config_.input_tensor_names[0]));
  auto *keypoints_1_buffer = static_cast<float *>(buffers.getHostBuffer(aliked_lightglue_config_.input_tensor_names[1]));
  auto *descriptors_0_buffer = static_cast<float *>(buffers.getHostBuffer(aliked_lightglue_config_.input_tensor_names[2]));
  auto *descriptors_1_buffer = static_cast<float *>(buffers.getHostBuffer(aliked_lightglue_config_.input_tensor_names[3]));

  PackAndNormalizeFeatures(features0.data(), features0.cols(), kFeatureRows, kDescriptorDim, image0_width, image0_height, keypoints_0_buffer, descriptors_0_buffer);
  PackAndNormalizeFeatures(features1.data(), features1.cols(), kFeatureRows, kDescriptorDim, image1_width, image1_height, keypoints_1_buffer, descriptors_1_buffer);
  return true;
}

bool AlikedLightGlue::ProcessOutput(const BufferManager &buffers, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score) {
  auto *output_scores = static_cast<float *>(buffers.getHostBuffer(aliked_lightglue_config_.output_tensor_names[0]));
  int scores_rows = keypoints_0_dims_.d[1] + 1;
  int scores_cols = keypoints_1_dims_.d[1] + 1;
  const float log_threshold = std::log(static_cast<float>(aliked_lightglue_config_.filter_threshold));
  MutualNearestMatches(output_scores, scores_rows, scores_cols, log_threshold, scores_rows - 1, scores_cols - 1, matches_index, matches_score);
  return true;
}

void AlikedLightGlue::SaveEngine() {
  if (aliked_lightglue_config_.engine_file.empty() || engine_ == nullptr) {
    return;
  }
  SaveSerializedEngine(*engine_, aliked_lightglue_config_.engine_file);
}

bool AlikedLightGlue::DeserializeEngine() {
  engine_ = LoadSerializedEngine(aliked_lightglue_config_.engine_file, "AlikedLightGlue");
  return engine_ != nullptr;
}

bool AlikedLightGlue::MatchingPoints(const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features0, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features1, std::vector<cv::DMatch> &matches,
                                     int image0_width, int image0_height, int image1_width, int image1_height, bool outlier_rejection) {
  matches.clear();

  Eigen::Matrix<int, Eigen::Dynamic, 2> matches_index;
  Eigen::Matrix<float, Eigen::Dynamic, 1> matches_score;
  if (!Infer(features0, features1, matches_index, matches_score, image0_width, image0_height, image1_width, image1_height)) {
    return false;
  }

  std::vector<cv::Point2f> points0, points1;
  for (size_t i = 0; i < matches_index.rows(); i++) {
    points0.emplace_back(features0(0, matches_index(i, 0)), features0(1, matches_index(i, 0)));
    points1.emplace_back(features1(0, matches_index(i, 1)), features1(1, matches_index(i, 1)));
    matches.emplace_back(matches_index(i, 0), matches_index(i, 1), matches_score(i));
  }

  if (outlier_rejection) {
    RejectFundamentalOutliers(points0, points1, matches);
  }

  return true;
}

}  // namespace viewfeat
