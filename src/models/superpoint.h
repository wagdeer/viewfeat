#ifndef VIEWFEAT_MODELS_SUPERPOINT_H_
#define VIEWFEAT_MODELS_SUPERPOINT_H_

#include <NvInfer.h>

#include <Eigen/Core>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

#include "buffers.h"
#include "runtime/config.h"

namespace viewfeat {

class SuperPoint {
 public:
  static constexpr int kDescriptorDim = 256;
  static constexpr int kFeatureRows = 2 + kDescriptorDim;

  explicit SuperPoint(SuperPointConfig superpoint_config);
  ~SuperPoint();

  bool Build();

  bool Infer(const cv::Mat &image, Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features, Eigen::Matrix<float, 1, Eigen::Dynamic> &features_score);

 private:
  void SaveEngine();

  bool DeserializeEngine();

  SuperPointConfig superpoint_config_;
  nvinfer1::Dims semi_dims_{};
  nvinfer1::Dims desc_dims_{};
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::unique_ptr<samplesCommon::BufferManager> buffers_;
  cudaStream_t stream_{nullptr};
  cv::Mat float_input_;
  int buffer_image_width_{0};
  int buffer_image_height_{0};
  std::vector<int> keypoints_x_;
  std::vector<int> keypoints_y_;
  std::vector<float> selected_scores_;

  bool EnsureBuffers(int image_width, int image_height);

  bool ProcessInput(const samplesCommon::BufferManager &buffers, const cv::Mat &image);

  bool ProcessOutput(const samplesCommon::BufferManager &buffers, Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features, Eigen::Matrix<float, 1, Eigen::Dynamic> &features_score);
};

}  // namespace viewfeat

#endif  // VIEWFEAT_MODELS_SUPERPOINT_H_
