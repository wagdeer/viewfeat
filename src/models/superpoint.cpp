#include "models/superpoint.h"

#include "runtime/engine_io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace viewfeat {

using sample::gLogger;
using samplesCommon::BufferManager;
using samplesCommon::SampleUniquePtr;

constexpr int SuperPoint::kDescriptorDim;
constexpr int SuperPoint::kFeatureRows;

namespace {

int clip(int val, int max_val) {
  if (val < 0) {
    return 0;
  }
  return std::min(val, max_val - 1);
}

struct KeypointNormalizeParams {
  float x_mul;
  float x_add;
  float y_mul;
  float y_add;
};

KeypointNormalizeParams make_keypoint_normalize_params(int h, int w, int s = 8) {
  const float shift = s / 2 - 0.5F;
  const float x_mul = 2.0F / (w * s - s / 2 - 0.5F);
  const float y_mul = 2.0F / (h * s - s / 2 - 0.5F);
  return {x_mul, -1.0F - shift * x_mul, y_mul, -1.0F - shift * y_mul};
}

inline void normalize_keypoint_coord_fast(int x, int y, float &out_x, float &out_y, const KeypointNormalizeParams &params) {
  out_x = x * params.x_mul + params.x_add;
  out_y = y * params.y_mul + params.y_add;
}

void grid_sample_and_normalize_column(const float *input, float gx, float gy, float *dest, int dim, int h, int w) {
  const float ix = ((gx + 1.0F) / 2.0F) * (w - 1);
  const float iy = ((gy + 1.0F) / 2.0F) * (h - 1);

  const int ix_nw = clip(static_cast<int>(std::floor(ix)), w);
  const int iy_nw = clip(static_cast<int>(std::floor(iy)), h);
  const int ix_ne = clip(ix_nw + 1, w);
  const int iy_ne = clip(iy_nw, h);
  const int ix_sw = clip(ix_nw, w);
  const int iy_sw = clip(iy_nw + 1, h);
  const int ix_se = clip(ix_nw + 1, w);
  const int iy_se = clip(iy_nw + 1, h);

  const float nw = (ix_se - ix) * (iy_se - iy);
  const float ne = (ix - ix_sw) * (iy_sw - iy);
  const float sw = (ix_ne - ix) * (iy - iy_ne);
  const float se = (ix - ix_nw) * (iy - iy_nw);

  const int offset_nw = iy_nw * w + ix_nw;
  const int offset_ne = iy_ne * w + ix_ne;
  const int offset_sw = iy_sw * w + ix_sw;
  const int offset_se = iy_se * w + ix_se;

  float sum = 0.0F;
  for (int i = 0; i < dim; ++i) {
    const float *plane = input + i * h * w;
    const float value = plane[offset_nw] * nw + plane[offset_ne] * ne + plane[offset_sw] * sw + plane[offset_se] * se;
    dest[i] = value;
    sum += value * value;
  }

  const float inv_norm = 1.0F / std::sqrt(sum);
  for (int i = 0; i < dim; ++i) {
    dest[i] *= inv_norm;
  }
}

void top_k_keypoints_flat(std::vector<int> &xs, std::vector<int> &ys, std::vector<float> &scores, int k) {
  if (k == -1 || k >= static_cast<int>(scores.size())) {
    return;
  }

  std::vector<size_t> indexes(scores.size());
  std::iota(indexes.begin(), indexes.end(), 0);
  std::nth_element(indexes.begin(), indexes.begin() + k, indexes.end(), [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });

  std::vector<int> top_x(k);
  std::vector<int> top_y(k);
  std::vector<float> top_scores(k);
  for (int i = 0; i < k; ++i) {
    top_x[i] = xs[indexes[i]];
    top_y[i] = ys[indexes[i]];
    top_scores[i] = scores[indexes[i]];
  }
  xs.swap(top_x);
  ys.swap(top_y);
  scores.swap(top_scores);
}

void select_keypoints(const float *output_score, int h, int w, float threshold, int border, int max_keypoints, std::vector<int> &xs, std::vector<int> &ys,
                      std::vector<float> &scores) {
  xs.clear();
  ys.clear();
  scores.clear();
  xs.reserve(512);
  ys.reserve(512);
  scores.reserve(512);

  const int map_size = h * w;
  for (int i = 0; i < map_size; ++i) {
    const float score = output_score[i];
    if (score <= threshold) {
      continue;
    }
    const int row = i / w;
    const int col = i % w;
    if (row < border || row >= h - border || col < border || col >= w - border) {
      continue;
    }
    xs.push_back(col);
    ys.push_back(row);
    scores.push_back(score);
  }

  top_k_keypoints_flat(xs, ys, scores, max_keypoints);
}

}  // namespace

SuperPoint::SuperPoint(SuperPointConfig superpoint_config) : superpoint_config_(std::move(superpoint_config)), engine_(nullptr), stream_(nullptr) {
  sample::setReportableSeverity(sample::Logger::Severity::kINTERNAL_ERROR);
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    stream_ = nullptr;
    std::cerr << "[SuperPoint] Failed to create CUDA stream." << std::endl;
  }
}

SuperPoint::~SuperPoint() {
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

bool SuperPoint::Build() {
  if (DeserializeEngine()) {
    buffers_.reset();
    buffer_image_width_ = 0;
    buffer_image_height_ = 0;
    return true;
  }
  engine_ = BuildEngineFromOnnx(superpoint_config_.onnx_file, superpoint_config_.dla_core, "SuperPoint");
  if (!engine_) {
    std::cerr << "[SuperPoint] Failed to build engine " << superpoint_config_.engine_file << "." << std::endl;
    return false;
  }
  SaveEngine();
  if (engine_->getNbIOTensors() != 3 || superpoint_config_.input_tensor_names.size() != 1 || superpoint_config_.output_tensor_names.size() != 2) {
    std::cerr << "[SuperPoint] Unexpected network IO count for " << superpoint_config_.engine_file << "." << std::endl;
    return false;
  }
  const nvinfer1::Dims input_dims = engine_->getTensorShape(superpoint_config_.input_tensor_names[0].c_str());
  semi_dims_ = engine_->getTensorShape(superpoint_config_.output_tensor_names[0].c_str());
  desc_dims_ = engine_->getTensorShape(superpoint_config_.output_tensor_names[1].c_str());
  if (input_dims.nbDims != 4 || semi_dims_.nbDims != 3 || desc_dims_.nbDims != 4) {
    std::cerr << "[SuperPoint] Unexpected network dimensions for " << superpoint_config_.engine_file << "." << std::endl;
    return false;
  }
  buffers_.reset();
  buffer_image_width_ = 0;
  buffer_image_height_ = 0;
  return true;
}

bool SuperPoint::EnsureBuffers(int image_width, int image_height) {
  if (buffers_ && buffer_image_width_ == image_width && buffer_image_height_ == image_height) {
    return true;
  }

  buffers_ = std::make_unique<BufferManager>(engine_, 0, context_.get(), samplesCommon::HostMemoryType::kPinned);
  if (!buffers_) {
    return false;
  }

  buffer_image_width_ = image_width;
  buffer_image_height_ = image_height;
  for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; ++i) {
    auto const tensor_name = engine_->getIOTensorName(i);
    if (!context_->setTensorAddress(tensor_name, buffers_->getDeviceBuffer(tensor_name))) {
      buffers_.reset();
      buffer_image_width_ = 0;
      buffer_image_height_ = 0;
      return false;
    }
  }
  return true;
}

bool SuperPoint::Infer(const cv::Mat &image, Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features, Eigen::Matrix<float, 1, Eigen::Dynamic> &features_score) {
  if (!context_) {
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      std::cerr << "[SuperPoint] Failed to create execution context for " << superpoint_config_.engine_file << "." << std::endl;
      return false;
    }
  }
  if (stream_ == nullptr) {
    std::cerr << "[SuperPoint] CUDA stream is missing." << std::endl;
    return false;
  }

  if (engine_->getNbIOTensors() != 3 || superpoint_config_.input_tensor_names.size() != 1 || superpoint_config_.output_tensor_names.size() != 2) {
    std::cerr << "[SuperPoint] Unexpected tensor layout for " << superpoint_config_.engine_file << "." << std::endl;
    return false;
  }

  auto const input_name = superpoint_config_.input_tensor_names[0].c_str();
  if (!context_->setInputShape(input_name, nvinfer1::Dims4(1, 1, image.rows, image.cols))) {
    return false;
  }

  if (!EnsureBuffers(image.cols, image.rows)) {
    return false;
  }

  semi_dims_ = context_->getTensorShape(superpoint_config_.output_tensor_names[0].c_str());
  desc_dims_ = context_->getTensorShape(superpoint_config_.output_tensor_names[1].c_str());

  if (!ProcessInput(*buffers_, image)) {
    return false;
  }

  buffers_->copyInputToDeviceAsync(stream_);
  if (!context_->enqueueV3(stream_)) {
    cudaStreamSynchronize(stream_);
    return false;
  }

  buffers_->copyOutputToHostAsync(stream_);
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    std::cerr << "[SuperPoint] cudaStreamSynchronize() failed." << std::endl;
    return false;
  }
  return ProcessOutput(*buffers_, features, features_score);
}

bool SuperPoint::ProcessInput(const BufferManager &buffers, const cv::Mat &image) {
  if (image.type() != CV_8UC1) {
    return false;
  }

  auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_config_.input_tensor_names[0]));
  const int rows = image.rows;
  const int cols = image.cols;
  const float inv255 = 1.0F / 255.0F;

  if (float_input_.rows != rows || float_input_.cols != cols || float_input_.type() != CV_32FC1) {
    float_input_.create(rows, cols, CV_32FC1);
  }
  image.convertTo(float_input_, CV_32FC1, inv255);
  std::memcpy(host_data_buffer, float_input_.ptr<float>(0), static_cast<size_t>(rows) * cols * sizeof(float));
  return true;
}

bool SuperPoint::ProcessOutput(const BufferManager &buffers, Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features, Eigen::Matrix<float, 1, Eigen::Dynamic> &features_score) {
  auto *output_score = static_cast<float *>(buffers.getHostBuffer(superpoint_config_.output_tensor_names[0]));
  auto *output_desc = static_cast<float *>(buffers.getHostBuffer(superpoint_config_.output_tensor_names[1]));
  if (desc_dims_.nbDims != 4 || desc_dims_.d[1] != kDescriptorDim) {
    const int channels = desc_dims_.nbDims > 1 ? desc_dims_.d[1] : -1;
    std::cerr << "[SuperPoint] Engine descriptor channels (" << channels << ") do not match " << kDescriptorDim << "." << std::endl;
    return false;
  }
  const int semi_feature_map_h = semi_dims_.d[1];
  const int semi_feature_map_w = semi_dims_.d[2];
  const int desc_feature_map_h = desc_dims_.d[2];
  const int desc_feature_map_w = desc_dims_.d[3];

  select_keypoints(output_score, semi_feature_map_h, semi_feature_map_w, static_cast<float>(superpoint_config_.keypoint_threshold), superpoint_config_.remove_borders,
                   superpoint_config_.max_keypoints, keypoints_x_, keypoints_y_, selected_scores_);

  const int num_keypoints = static_cast<int>(keypoints_x_.size());
  features.resize(kFeatureRows, num_keypoints);
  features_score.resize(1, num_keypoints);
  if (num_keypoints > 0) {
    std::memcpy(features_score.data(), selected_scores_.data(), num_keypoints * sizeof(float));
  }

  const KeypointNormalizeParams normalize_params = make_keypoint_normalize_params(desc_feature_map_h, desc_feature_map_w);
  for (int i = 0; i < num_keypoints; ++i) {
    float *feature_col = features.data() + i * kFeatureRows;
    feature_col[0] = static_cast<float>(keypoints_x_[i]);
    feature_col[1] = static_cast<float>(keypoints_y_[i]);

    float gx = 0.0F;
    float gy = 0.0F;
    normalize_keypoint_coord_fast(keypoints_x_[i], keypoints_y_[i], gx, gy, normalize_params);
    grid_sample_and_normalize_column(output_desc, gx, gy, feature_col + 2, kDescriptorDim, desc_feature_map_h, desc_feature_map_w);
  }
  return true;
}

void SuperPoint::SaveEngine() {
  if (superpoint_config_.engine_file.empty() || engine_ == nullptr) {
    return;
  }
  SaveSerializedEngine(*engine_, superpoint_config_.engine_file);
}

bool SuperPoint::DeserializeEngine() {
  engine_ = LoadSerializedEngine(superpoint_config_.engine_file, "SuperPoint");
  return engine_ != nullptr;
}

}  // namespace viewfeat
