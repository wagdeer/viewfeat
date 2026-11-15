#include "models/aliked.h"

#include "runtime/engine_io.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace viewfeat {

using sample::gLogger;
using samplesCommon::BufferManager;
using samplesCommon::SampleUniquePtr;

constexpr int Aliked::kDescriptorDim;
constexpr int Aliked::kFeatureRows;

Aliked::Aliked(AlikedConfig aliked_config) : aliked_config_(std::move(aliked_config)), engine_(nullptr), stream_(nullptr) {
  sample::setReportableSeverity(sample::Logger::Severity::kINTERNAL_ERROR);
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    stream_ = nullptr;
    std::cerr << "[Aliked] Failed to create CUDA stream." << std::endl;
  }
}

Aliked::~Aliked() {
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

bool Aliked::Build() {
  if (DeserializeEngine()) {
    return true;
  }

  // TensorRT 10.9 levels 4 and 5 fuse the scale-8 upsample. The score map stays
  // 36 rows tall (H/8) and keypoint y only covers the top of the image.
  engine_ = BuildEngineFromOnnx(aliked_config_.onnx_file, aliked_config_.dla_core, "Aliked", 3);
  if (!engine_) {
    std::cerr << "[Aliked] Failed to build engine " << aliked_config_.engine_file << "." << std::endl;
    return false;
  }

  SaveEngine();
  if (engine_->getNbIOTensors() != 4 || aliked_config_.input_tensor_names.size() != 1 || aliked_config_.output_tensor_names.size() != 3) {
    std::cerr << "[Aliked] Unexpected network IO count for " << aliked_config_.engine_file << "." << std::endl;
    return false;
  }
  keypoints_dims_ = engine_->getTensorShape(aliked_config_.output_tensor_names[0].c_str());
  descriptors_dims_ = engine_->getTensorShape(aliked_config_.output_tensor_names[1].c_str());
  return true;
}

bool Aliked::Infer(const cv::Mat &image, Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features, Eigen::Matrix<float, 1, Eigen::Dynamic> &features_score) {
  if (!context_) {
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      std::cerr << "[Aliked] Failed to create execution context for " << aliked_config_.engine_file << "." << std::endl;
      return false;
    }
  }
  if (stream_ == nullptr) {
    std::cerr << "[Aliked] CUDA stream is missing." << std::endl;
    return false;
  }

  if (engine_->getNbIOTensors() != 4 || aliked_config_.input_tensor_names.size() != 1 || aliked_config_.output_tensor_names.size() != 3) {
    std::cerr << "[Aliked] Unexpected tensor layout for " << aliked_config_.engine_file << "." << std::endl;
    return false;
  }

  auto const input_name = aliked_config_.input_tensor_names[0].c_str();
  if (!context_->setInputShape(input_name, nvinfer1::Dims4(1, 3, image.rows, image.cols))) {
    return false;
  }

  if (!EnsureBuffers(image.cols, image.rows)) {
    return false;
  }

  keypoints_dims_ = context_->getTensorShape(aliked_config_.output_tensor_names[0].c_str());
  descriptors_dims_ = context_->getTensorShape(aliked_config_.output_tensor_names[1].c_str());

  if (!ProcessInput(*buffers_, image)) {
    return false;
  }
  buffers_->copyInputToDeviceAsync(stream_);
  if (!context_->enqueueV3(stream_)) {
    cudaStreamSynchronize(stream_);
    std::cerr << "[Aliked] enqueueV3() failed." << std::endl;
    return false;
  }
  buffers_->copyOutputToHostAsync(stream_);
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    std::cerr << "[Aliked] cudaStreamSynchronize() failed." << std::endl;
    return false;
  }
  if (!ProcessOutput(*buffers_, features, features_score, image.cols, image.rows)) {
    return false;
  }
  return true;
}

bool Aliked::EnsureBuffers(int image_width, int image_height) {
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

bool Aliked::ProcessInput(const BufferManager &buffers, const cv::Mat &image) {
  auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer(aliked_config_.input_tensor_names[0]));
  int conversion = -1;
  if (image.channels() == 3) {
    conversion = cv::COLOR_BGR2RGB;
  } else if (image.channels() == 4) {
    conversion = cv::COLOR_BGRA2RGB;
  } else if (image.channels() == 1) {
    conversion = cv::COLOR_GRAY2RGB;
  } else {
    std::cerr << "[Aliked] Unsupported channel count: " << image.channels() << "." << std::endl;
    return false;
  }
  cv::cvtColor(image, rgb_image_, conversion);

  const int rows = rgb_image_.rows;
  const int cols = rgb_image_.cols;
  const int channel_size = rows * cols;
  const float inv255 = 1.0F / 255.0F;

  for (int row = 0; row < rows; ++row) {
    const cv::Vec3b *row_ptr = rgb_image_.ptr<cv::Vec3b>(row);
    const int row_offset = row * cols;
    for (int col = 0; col < cols; ++col) {
      const cv::Vec3b &pixel = row_ptr[col];
      const int idx = row_offset + col;
      host_data_buffer[idx] = pixel[0] * inv255;
      host_data_buffer[channel_size + idx] = pixel[1] * inv255;
      host_data_buffer[2 * channel_size + idx] = pixel[2] * inv255;
    }
  }
  return true;
}

bool Aliked::ProcessOutput(const BufferManager &buffers, Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features, Eigen::Matrix<float, 1, Eigen::Dynamic> &features_score, int image_width,
                            int image_height) {
  auto *output_keypoints = static_cast<float *>(buffers.getHostBuffer(aliked_config_.output_tensor_names[0]));
  auto *output_descriptors = static_cast<float *>(buffers.getHostBuffer(aliked_config_.output_tensor_names[1]));
  auto *output_scores = static_cast<float *>(buffers.getHostBuffer(aliked_config_.output_tensor_names[2]));

  int num_keypoints = aliked_config_.top_k;
  if (keypoints_dims_.nbDims == 2) {
    num_keypoints = keypoints_dims_.d[0];
  } else if (keypoints_dims_.nbDims == 3) {
    num_keypoints = keypoints_dims_.d[1];
  }
  if (num_keypoints <= 0) {
    return false;
  }

  const int channel = descriptors_dims_.nbDims >= 1 ? descriptors_dims_.d[descriptors_dims_.nbDims - 1] : -1;
  long long described_points = 1;
  bool layout_ok = descriptors_dims_.nbDims >= 2 && channel == kDescriptorDim;
  if (layout_ok) {
    for (int i = 0; i < descriptors_dims_.nbDims - 1; ++i) {
      if (descriptors_dims_.d[i] <= 0) {
        layout_ok = false;
        break;
      }
      described_points *= descriptors_dims_.d[i];
    }
  }
  if (!layout_ok || described_points < num_keypoints) {
    std::cerr << "[Aliked] Engine descriptor channels (" << channel << ") do not match " << kDescriptorDim << "." << std::endl;
    return false;
  }

  features.resize(kFeatureRows, num_keypoints);
  features_score.resize(1, num_keypoints);
  const float width_scale = (image_width - 1) * 0.5F;
  const float height_scale = (image_height - 1) * 0.5F;

  std::memcpy(features_score.data(), output_scores, num_keypoints * sizeof(float));
  for (int col = 0; col < num_keypoints; ++col) {
    const float x_norm = output_keypoints[col * 2];
    const float y_norm = output_keypoints[col * 2 + 1];
    features(0, col) = (x_norm + 1.0F) * width_scale;
    features(1, col) = (y_norm + 1.0F) * height_scale;
    std::memcpy(features.data() + 2 + col * kFeatureRows, output_descriptors + col * kDescriptorDim, kDescriptorDim * sizeof(float));
  }
  return true;
}

void Aliked::SaveEngine() {
  if (aliked_config_.engine_file.empty() || engine_ == nullptr) {
    return;
  }
  SaveSerializedEngine(*engine_, aliked_config_.engine_file);
}

bool Aliked::DeserializeEngine() {
  engine_ = LoadSerializedEngine(aliked_config_.engine_file, "Aliked");
  return engine_ != nullptr;
}

}  // namespace viewfeat
