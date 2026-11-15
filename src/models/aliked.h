#ifndef VIEWFEAT_MODELS_ALIKED_H_
#define VIEWFEAT_MODELS_ALIKED_H_

#include <NvInfer.h>

#include <Eigen/Core>
#include <memory>
#include <opencv2/core/mat.hpp>

#include "buffers.h"
#include "runtime/config.h"

namespace viewfeat {

class Aliked {
 public:
  static constexpr int kDescriptorDim = 128;
  static constexpr int kFeatureRows = 2 + kDescriptorDim;

  explicit Aliked(AlikedConfig aliked_config);
  ~Aliked();

  bool Build();

  bool Infer(const cv::Mat &image, Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features, Eigen::Matrix<float, 1, Eigen::Dynamic> &features_score);

 private:
  void SaveEngine();

  bool DeserializeEngine();

  AlikedConfig aliked_config_;
  nvinfer1::Dims keypoints_dims_{};
  nvinfer1::Dims descriptors_dims_{};
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::unique_ptr<samplesCommon::BufferManager> buffers_;
  int buffer_image_width_{0};
  int buffer_image_height_{0};
  cv::Mat rgb_image_;
  cudaStream_t stream_{nullptr};

  bool EnsureBuffers(int image_width, int image_height);

  bool ProcessInput(const samplesCommon::BufferManager &buffers, const cv::Mat &image);

  bool ProcessOutput(const samplesCommon::BufferManager &buffers, Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features, Eigen::Matrix<float, 1, Eigen::Dynamic> &features_score,
                      int image_width, int image_height);
};

}  // namespace viewfeat

#endif  // VIEWFEAT_MODELS_ALIKED_H_
