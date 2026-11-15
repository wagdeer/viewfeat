#ifndef VIEWFEAT_MODELS_PLACE_DESCRIPTOR_H_
#define VIEWFEAT_MODELS_PLACE_DESCRIPTOR_H_

#include <NvInfer.h>

#include <Eigen/Core>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "buffers.h"
#include "runtime/config.h"

namespace viewfeat {

// Shared TensorRT runtime for a single global place descriptor.
// CosPlace and EigenPlaces use the same input layout, ImageNet preprocessing, and L2 output.
class PlaceDescriptor {
 public:
  PlaceDescriptor(PlaceDescriptorConfig config, std::string label);
  ~PlaceDescriptor();

  bool Build();

  bool Infer(const cv::Mat &image, Eigen::VectorXf &descriptor);

 private:
  void SaveEngine();

  bool DeserializeEngine();

  bool EnsureBuffers();

  bool ProcessInput(const samplesCommon::BufferManager &buffers, const cv::Mat &image);

  bool ProcessOutput(const samplesCommon::BufferManager &buffers, Eigen::VectorXf &descriptor);

  PlaceDescriptorConfig config_;
  std::string label_;
  int descriptor_dim_{0};
  nvinfer1::Dims output_dims_{};
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::unique_ptr<samplesCommon::BufferManager> buffers_;
  cudaStream_t stream_{nullptr};
};

}  // namespace viewfeat

#endif  // VIEWFEAT_MODELS_PLACE_DESCRIPTOR_H_
