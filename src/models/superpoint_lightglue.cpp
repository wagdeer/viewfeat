#include "models/superpoint_lightglue.h"

#include "models/lightglue_postprocess.h"
#include "models/outlier_rejection.h"
#include "runtime/engine_io.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>

namespace viewfeat {

using sample::gLogger;
using samplesCommon::BufferManager;
using samplesCommon::SampleUniquePtr;

constexpr int SuperPointLightGlue::kDescriptorDim;
constexpr int SuperPointLightGlue::kFeatureRows;

SuperPointLightGlue::SuperPointLightGlue(SuperPointLightGlueConfig superpoint_lightglue_config)
    : superpoint_lightglue_config_(std::move(superpoint_lightglue_config)), engine_(nullptr), stream_(nullptr) {
  sample::setReportableSeverity(sample::Logger::Severity::kINTERNAL_ERROR);
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    stream_ = nullptr;
    std::cerr << "[SuperPointLightGlue] Failed to create CUDA stream." << std::endl;
  }
}

SuperPointLightGlue::~SuperPointLightGlue() {
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

bool SuperPointLightGlue::Build() {
  if (DeserializeEngine()) {
    auto const keypoints_0_name = superpoint_lightglue_config_.input_tensor_names[0].c_str();
    auto const keypoints_1_name = superpoint_lightglue_config_.input_tensor_names[1].c_str();
    keypoints_0_dims_ = engine_->getTensorShape(keypoints_0_name);
    keypoints_1_dims_ = engine_->getTensorShape(keypoints_1_name);
    buffers_.reset();
    buffer_keypoints0_ = 0;
    buffer_keypoints1_ = 0;
    return true;
  }

  engine_ = BuildEngineFromOnnx(superpoint_lightglue_config_.onnx_file, superpoint_lightglue_config_.dla_core, "SuperPointLightGlue");
  if (!engine_) {
    std::cerr << "[SuperPointLightGlue] Failed to build engine " << superpoint_lightglue_config_.engine_file << "." << std::endl;
    return false;
  }

  SaveEngine();

  if (engine_->getNbIOTensors() != 5 || superpoint_lightglue_config_.input_tensor_names.size() != 4 ||
      superpoint_lightglue_config_.output_tensor_names.size() != 1) {
    std::cerr << "[SuperPointLightGlue] Expected 4 network inputs for " << superpoint_lightglue_config_.engine_file << "." << std::endl;
    return false;
  }
  keypoints_0_dims_ = engine_->getTensorShape(superpoint_lightglue_config_.input_tensor_names[0].c_str());
  keypoints_1_dims_ = engine_->getTensorShape(superpoint_lightglue_config_.input_tensor_names[1].c_str());
  buffers_.reset();
  buffer_keypoints0_ = 0;
  buffer_keypoints1_ = 0;
  return true;
}

void SuperPointLightGlue::ReadKeypointProfile() {
  if (profile_ready_) {
    return;
  }
  profile_ready_ = true;
  profile_dynamic_ = false;
  const int profiles = engine_->getNbOptimizationProfiles();
  if (profiles < 1 || superpoint_lightglue_config_.input_tensor_names.size() < 4) {
    std::cout << "[SuperPointLightGlue] keypoint profile fixed (no optimization profile)" << std::endl;
    return;
  }

  const char *first = superpoint_lightglue_config_.input_tensor_names[0].c_str();
  const nvinfer1::Dims min_dims = engine_->getProfileShape(first, 0, nvinfer1::OptProfileSelector::kMIN);
  const nvinfer1::Dims max_dims = engine_->getProfileShape(first, 0, nvinfer1::OptProfileSelector::kMAX);
  if (min_dims.nbDims < 2 || max_dims.nbDims < 2 || min_dims.d[1] <= 0 || max_dims.d[1] <= 0) {
    std::cout << "[SuperPointLightGlue] keypoint profile fixed" << std::endl;
    return;
  }
  profile_min_ = min_dims.d[1];
  profile_max_ = max_dims.d[1];
  bool same = true;
  for (size_t i = 1; i < 4; ++i) {
    const char *name = superpoint_lightglue_config_.input_tensor_names[i].c_str();
    const nvinfer1::Dims other_min = engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN);
    const nvinfer1::Dims other_max = engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
    if (other_min.nbDims < 2 || other_max.nbDims < 2 || other_min.d[1] != profile_min_ || other_max.d[1] != profile_max_) {
      same = false;
      break;
    }
  }
  profile_dynamic_ = same && profile_min_ != profile_max_;
  std::cout << "[SuperPointLightGlue] keypoint profile " << (profile_dynamic_ ? "dynamic" : "fixed") << " min=" << profile_min_ << " max=" << profile_max_ << std::endl;
}

int SuperPointLightGlue::KeypointCount(int available) const {
  if (!profile_dynamic_) {
    return superpoint_lightglue_config_.num_keypoints;
  }
  int count = available;
  if (count < profile_min_) {
    count = profile_min_;
  }
  if (count > profile_max_) {
    count = profile_max_;
  }
  return count;
}

bool SuperPointLightGlue::EnsureBuffers(int keypoints0, int keypoints1) {
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

bool SuperPointLightGlue::Infer(const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features0, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features1,
                                Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score, int image0_width, int image0_height,
                                int image1_width, int image1_height) {
  if (!context_) {
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      std::cerr << "[SuperPointLightGlue] Failed to create execution context for " << superpoint_lightglue_config_.engine_file << "." << std::endl;
      return false;
    }
  }
  if (stream_ == nullptr) {
    std::cerr << "[SuperPointLightGlue] CUDA stream is missing." << std::endl;
    return false;
  }

  if (engine_->getNbIOTensors() != 5 || superpoint_lightglue_config_.input_tensor_names.size() != 4 ||
      superpoint_lightglue_config_.output_tensor_names.size() != 1) {
    std::cerr << "[SuperPointLightGlue] Unexpected tensor layout for " << superpoint_lightglue_config_.engine_file << "." << std::endl;
    return false;
  }

  auto const keypoints_0_name = superpoint_lightglue_config_.input_tensor_names[0].c_str();
  auto const keypoints_1_name = superpoint_lightglue_config_.input_tensor_names[1].c_str();
  auto const descriptors_0_name = superpoint_lightglue_config_.input_tensor_names[2].c_str();
  auto const descriptors_1_name = superpoint_lightglue_config_.input_tensor_names[3].c_str();

  ReadKeypointProfile();
  input_keypoints0_ = KeypointCount(features0.cols());
  input_keypoints1_ = KeypointCount(features1.cols());
  valid_keypoints0_ = std::min<int>(features0.cols(), input_keypoints0_);
  valid_keypoints1_ = std::min<int>(features1.cols(), input_keypoints1_);
  const int input_keypoints0 = input_keypoints0_;
  const int input_keypoints1 = input_keypoints1_;

  if (!context_->setInputShape(keypoints_0_name, nvinfer1::Dims3(1, input_keypoints0, 2))) {
    return false;
  }
  if (!context_->setInputShape(keypoints_1_name, nvinfer1::Dims3(1, input_keypoints1, 2))) {
    return false;
  }
  if (!context_->setInputShape(descriptors_0_name, nvinfer1::Dims3(1, input_keypoints0, kDescriptorDim))) {
    return false;
  }
  if (!context_->setInputShape(descriptors_1_name, nvinfer1::Dims3(1, input_keypoints1, kDescriptorDim))) {
    return false;
  }

  if (!EnsureBuffers(input_keypoints0, input_keypoints1)) {
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
    std::cerr << "[SuperPointLightGlue] cudaStreamSynchronize() failed." << std::endl;
    return false;
  }

  if (!ProcessOutput(*buffers_, matches_index, matches_score)) {
    return false;
  }

  return true;
}

bool SuperPointLightGlue::ProcessInput(const BufferManager &buffers, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features0, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features1,
                                        int image0_width, int image0_height, int image1_width, int image1_height) {
  auto *keypoints_0_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[0]));
  auto *keypoints_1_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[1]));
  auto *descriptors_0_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[2]));
  auto *descriptors_1_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[3]));

  const int feature_rows = features0.rows();
  if (input_keypoints0_ > valid_keypoints0_) {
    std::fill(keypoints_0_buffer, keypoints_0_buffer + input_keypoints0_ * 2, 0.0F);
    std::fill(descriptors_0_buffer, descriptors_0_buffer + input_keypoints0_ * kDescriptorDim, 0.0F);
  }
  if (input_keypoints1_ > valid_keypoints1_) {
    std::fill(keypoints_1_buffer, keypoints_1_buffer + input_keypoints1_ * 2, 0.0F);
    std::fill(descriptors_1_buffer, descriptors_1_buffer + input_keypoints1_ * kDescriptorDim, 0.0F);
  }

  PackAndNormalizeFeatures(features0.data(), valid_keypoints0_, feature_rows, kDescriptorDim, image0_width, image0_height, keypoints_0_buffer, descriptors_0_buffer);
  PackAndNormalizeFeatures(features1.data(), valid_keypoints1_, feature_rows, kDescriptorDim, image1_width, image1_height, keypoints_1_buffer, descriptors_1_buffer);
  return true;
}

bool SuperPointLightGlue::ProcessOutput(const BufferManager &buffers, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score) {
  auto *output_scores = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.output_tensor_names[0]));
  int scores_rows = keypoints_0_dims_.d[1] + 1;
  int scores_cols = keypoints_1_dims_.d[1] + 1;
  const float log_threshold = std::log(static_cast<float>(superpoint_lightglue_config_.filter_threshold));
  MutualNearestMatches(output_scores, scores_rows, scores_cols, log_threshold, valid_keypoints0_, valid_keypoints1_, matches_index, matches_score);
  return true;
}

void SuperPointLightGlue::SaveEngine() {
  if (superpoint_lightglue_config_.engine_file.empty() || engine_ == nullptr) {
    return;
  }
  SaveSerializedEngine(*engine_, superpoint_lightglue_config_.engine_file);
}

bool SuperPointLightGlue::DeserializeEngine() {
  engine_ = LoadSerializedEngine(superpoint_lightglue_config_.engine_file, "SuperPointLightGlue");
  return engine_ != nullptr;
}

bool SuperPointLightGlue::MatchingPoints(const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features0, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features1, std::vector<cv::DMatch> &matches,
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
