#include "models/place_descriptor.h"

#include "runtime/engine_io.h"

#include <cstring>
#include <iostream>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace viewfeat {

using samplesCommon::BufferManager;
using samplesCommon::HostMemoryType;

namespace {

constexpr float kImageNetMean[3] = {0.485F, 0.456F, 0.406F};
constexpr float kImageNetStd[3] = {0.229F, 0.224F, 0.225F};

void L2Normalize(Eigen::VectorXf &vec) {
  const float norm = vec.norm();
  if (norm > 0.0F) {
    vec /= norm;
  }
}

int64_t TensorVolume(const nvinfer1::Dims &dims) {
  int64_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      return -1;
    }
    volume *= dims.d[i];
  }
  return volume;
}

}  // namespace

PlaceDescriptor::PlaceDescriptor(PlaceDescriptorConfig config, std::string label)
    : config_(std::move(config)), label_(std::move(label)), engine_(nullptr), stream_(nullptr) {
  sample::setReportableSeverity(sample::Logger::Severity::kINTERNAL_ERROR);
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    stream_ = nullptr;
    std::cerr << "[" << label_ << "] Failed to create CUDA stream." << std::endl;
  }
}

PlaceDescriptor::~PlaceDescriptor() {
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

bool PlaceDescriptor::Build() {
  if (DeserializeEngine()) {
    std::cout << "[" << label_ << "] Loaded TensorRT engine: " << config_.engine_file << "." << std::endl;
    return true;
  }

  engine_ = BuildEngineFromOnnx(config_.onnx_file, config_.dla_core, label_.c_str());
  if (!engine_) {
    std::cerr << "[" << label_ << "] Failed to build engine " << config_.engine_file << "." << std::endl;
    return false;
  }

  SaveEngine();
  if (engine_->getNbIOTensors() != 2 || config_.input_tensor_names.size() != 1 || config_.output_tensor_names.size() != 1) {
    std::cerr << "[" << label_ << "] Unexpected network IO count for " << config_.engine_file << "." << std::endl;
    return false;
  }
  output_dims_ = engine_->getTensorShape(config_.output_tensor_names[0].c_str());
  descriptor_dim_ = static_cast<int>(TensorVolume(output_dims_));
  std::cout << "[" << label_ << "] Built TensorRT engine from ONNX: " << config_.engine_file << "." << std::endl;
  return true;
}

bool PlaceDescriptor::Infer(const cv::Mat &image, Eigen::VectorXf &descriptor) {
  if (image.empty() || engine_ == nullptr) {
    return false;
  }

  if (!context_) {
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      std::cerr << "[" << label_ << "] Failed to create execution context for " << config_.engine_file << "." << std::endl;
      return false;
    }
  }
  if (stream_ == nullptr) {
    std::cerr << "[" << label_ << "] CUDA stream is missing." << std::endl;
    return false;
  }

  if (config_.input_tensor_names.size() != 1 || config_.output_tensor_names.size() != 1 || engine_->getNbIOTensors() != 2) {
    std::cerr << "[" << label_ << "] Unexpected tensor layout for " << config_.engine_file << "." << std::endl;
    return false;
  }

  const int input_size = config_.input_size;
  auto const input_name = config_.input_tensor_names[0].c_str();
  const nvinfer1::Dims4 input_shape(1, 3, input_size, input_size);
  if (!context_->setInputShape(input_name, input_shape)) {
    std::cerr << "[" << label_ << "] setInputShape() failed for fixed input " << input_size << "x" << input_size << "." << std::endl;
    return false;
  }

  if (!EnsureBuffers()) {
    return false;
  }

  output_dims_ = context_->getTensorShape(config_.output_tensor_names[0].c_str());
  const int64_t output_count = TensorVolume(output_dims_);
  if (output_count <= 0) {
    std::cerr << "[" << label_ << "] Invalid output tensor shape." << std::endl;
    return false;
  }
  descriptor_dim_ = static_cast<int>(output_count);

  if (!ProcessInput(*buffers_, image)) {
    return false;
  }
  buffers_->copyInputToDeviceAsync(stream_);
  if (!context_->enqueueV3(stream_)) {
    cudaStreamSynchronize(stream_);
    std::cerr << "[" << label_ << "] enqueueV3() failed." << std::endl;
    return false;
  }
  buffers_->copyOutputToHostAsync(stream_);
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    std::cerr << "[" << label_ << "] cudaStreamSynchronize() failed." << std::endl;
    return false;
  }
  if (!ProcessOutput(*buffers_, descriptor)) {
    return false;
  }
  return true;
}

bool PlaceDescriptor::EnsureBuffers() {
  if (buffers_) {
    return true;
  }

  buffers_ = std::make_unique<BufferManager>(engine_, 0, context_.get(), HostMemoryType::kPinned);
  if (!buffers_) {
    return false;
  }
  for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; ++i) {
    auto const tensor_name = engine_->getIOTensorName(i);
    if (!context_->setTensorAddress(tensor_name, buffers_->getDeviceBuffer(tensor_name))) {
      buffers_.reset();
      return false;
    }
  }
  return true;
}

bool PlaceDescriptor::ProcessInput(const BufferManager &buffers, const cv::Mat &image) {
  cv::Mat rgb;
  if (image.channels() == 1) {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
  } else {
    std::cerr << "[" << label_ << "] Unsupported channel count: " << image.channels() << "." << std::endl;
    return false;
  }

  const int input_size = config_.input_size;
  cv::resize(rgb, rgb, cv::Size(input_size, input_size), 0.0, 0.0, cv::INTER_LINEAR);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer(config_.input_tensor_names[0]));
  const int channel_size = input_size * input_size;
  for (int y = 0; y < input_size; ++y) {
    for (int x = 0; x < input_size; ++x) {
      const cv::Vec3f pixel = rgb.at<cv::Vec3f>(y, x);
      const int idx = y * input_size + x;
      for (int c = 0; c < 3; ++c) {
        host_data_buffer[c * channel_size + idx] = (pixel[c] - kImageNetMean[c]) / kImageNetStd[c];
      }
    }
  }
  return true;
}

bool PlaceDescriptor::ProcessOutput(const BufferManager &buffers, Eigen::VectorXf &descriptor) {
  const auto *host_output = static_cast<const float *>(buffers.getHostBuffer(config_.output_tensor_names[0]));
  if (host_output == nullptr || descriptor_dim_ <= 0) {
    return false;
  }

  descriptor.resize(descriptor_dim_);
  std::memcpy(descriptor.data(), host_output, static_cast<size_t>(descriptor_dim_) * sizeof(float));
  L2Normalize(descriptor);
  return true;
}

void PlaceDescriptor::SaveEngine() {
  if (config_.engine_file.empty() || engine_ == nullptr) {
    return;
  }
  SaveSerializedEngine(*engine_, config_.engine_file);
}

bool PlaceDescriptor::DeserializeEngine() {
  engine_ = LoadSerializedEngine(config_.engine_file, label_.c_str());
  if (!engine_) {
    return false;
  }

  if (!context_) {
    context_.reset(engine_->createExecutionContext());
  }
  if (!context_) {
    std::cerr << "[" << label_ << "] Failed to create execution context for " << config_.engine_file << "." << std::endl;
    engine_.reset();
    return false;
  }

  if (config_.input_tensor_names.size() != 1 || config_.output_tensor_names.size() != 1) {
    std::cerr << "[" << label_ << "] Unexpected tensor names for " << config_.engine_file << "." << std::endl;
    engine_.reset();
    context_.reset();
    return false;
  }

  const int input_size = config_.input_size;
  auto const input_name = config_.input_tensor_names[0].c_str();
  if (!context_->setInputShape(input_name, nvinfer1::Dims4(1, 3, input_size, input_size))) {
    std::cerr << "[" << label_ << "] setInputShape() failed while loading engine." << std::endl;
    engine_.reset();
    context_.reset();
    return false;
  }

  output_dims_ = context_->getTensorShape(config_.output_tensor_names[0].c_str());
  descriptor_dim_ = static_cast<int>(TensorVolume(output_dims_));
  if (descriptor_dim_ != config_.descriptor_dim) {
    std::cerr << "[" << label_ << "] Engine descriptor dim (" << descriptor_dim_ << ") does not match config ("
              << config_.descriptor_dim << ")." << std::endl;
    engine_.reset();
    context_.reset();
    return false;
  }

  return true;
}

}  // namespace viewfeat
