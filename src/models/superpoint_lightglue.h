#ifndef VIEWFEAT_MODELS_SUPERPOINT_LIGHTGLUE_H_
#define VIEWFEAT_MODELS_SUPERPOINT_LIGHTGLUE_H_

#include <NvInfer.h>

#include <Eigen/Core>
#include <memory>
#include <opencv2/core/types.hpp>
#include <vector>

#include "buffers.h"
#include "runtime/config.h"

namespace viewfeat {

class SuperPointLightGlue {
 public:
  static constexpr int kDescriptorDim = 256;
  static constexpr int kFeatureRows = 2 + kDescriptorDim;

  explicit SuperPointLightGlue(SuperPointLightGlueConfig superpoint_lightglue_config);
  ~SuperPointLightGlue();

  bool Build();

  bool Infer(const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features0, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features1,
             Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score, int image0_width, int image0_height, int image1_width,
             int image1_height);

  bool MatchingPoints(const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features0, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features1, std::vector<cv::DMatch> &matches,
                      int image0_width, int image0_height, int image1_width, int image1_height, bool outlier_rejection = false);

 private:
  void SaveEngine();

  bool DeserializeEngine();

  SuperPointLightGlueConfig superpoint_lightglue_config_;
  int valid_keypoints0_{};
  int valid_keypoints1_{};
  int input_keypoints0_{};
  int input_keypoints1_{};
  int buffer_keypoints0_{0};
  int buffer_keypoints1_{0};
  bool profile_ready_{false};
  bool profile_dynamic_{false};
  int profile_min_{0};
  int profile_max_{0};

  nvinfer1::Dims keypoints_0_dims_{};
  nvinfer1::Dims keypoints_1_dims_{};

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::unique_ptr<samplesCommon::BufferManager> buffers_;
  cudaStream_t stream_{nullptr};

  void ReadKeypointProfile();

  int KeypointCount(int available) const;

  bool EnsureBuffers(int keypoints0, int keypoints1);

  bool ProcessInput(const samplesCommon::BufferManager &buffers, const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features0,
                     const Eigen::Matrix<float, kFeatureRows, Eigen::Dynamic> &features1,
                     int image0_width, int image0_height, int image1_width, int image1_height);

  bool ProcessOutput(const samplesCommon::BufferManager &buffers, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score);
};

}  // namespace viewfeat

#endif  // VIEWFEAT_MODELS_SUPERPOINT_LIGHTGLUE_H_
